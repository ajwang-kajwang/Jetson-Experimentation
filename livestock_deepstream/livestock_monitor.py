#!/usr/bin/env python3
"""
Production Livestock Monitoring System
"""
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import json
import signal
import time
import argparse
import threading
from collections import defaultdict, deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# GPIO setup 
try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
    GPIO.setmode(GPIO.BOARD)
    
    COW_GPIO_PIN = 29
    SHEEP_GPIO_PIN = 31
    PIG_GPIO_PIN = 33
    ALERT_GPIO_PIN = 32
    
    for pin in (COW_GPIO_PIN, SHEEP_GPIO_PIN, PIG_GPIO_PIN):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
except:
    HAS_GPIO = False
    print("Benchmark Mode")

# Metrics for dashboard
metrics = {
    "total": 0,
    "cows": 0,
    "pigs": 0, 
    "sheep": 0,
    "fps": 0,
    "frame": 0,
    "largest_animal": "none",
    "alerts": []
}

# FPS tracking 
_fps_ts = defaultdict(lambda: deque(maxlen=60))

# Pipeline control
shutting_down = {"flag": False}

class LivestockMonitor:
    """Production-ready Livestock Monitor"""
    
    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self.loop = None
        self.probe_ids = []
        
    def make(self, factory, name):
        """Create elements"""
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(f"Could not create element: {factory} ({name})")
        return elem
    
    def link_many(self, *elems):
        """Link elements"""
        for a, b in zip(elems, elems[1:]):
            if not a.link(b):
                raise RuntimeError(f"Failed to link {a.name} -> {b.name}")
        return True
    
    def create_source_bin(self, index, uri_or_device):
        """Create source bin for either IP camera or CSI camera"""
        bin_name = f"source-bin-{index}"
        
        if uri_or_device.startswith("csi"):
            # CSI camera
            print(f"Creating CSI camera source {index}")
            
            src = self.make("nvarguscamerasrc", f"csi-src-{index}")
            caps1 = self.make("capsfilter", f"csi-caps1-{index}")
            caps1.set_property("caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=30/1"
            ))
            nvconv = self.make("nvvideoconvert", f"csi-nvconv-{index}")
            caps2 = self.make("capsfilter", f"csi-caps2-{index}")
            caps2.set_property("caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12"
            ))
            
            # Create bin and add elements
            source_bin = Gst.Bin.new(bin_name)
            for elem in (src, caps1, nvconv, caps2):
                source_bin.add(elem)
            self.link_many(src, caps1, nvconv, caps2)
            
            # Add ghost pad
            pad = caps2.get_static_pad("src")
            ghost_pad = Gst.GhostPad.new("src", pad)
            source_bin.add_pad(ghost_pad)
            
        else:
            # IP camera
            print(f"Creating IP camera source {index}: {uri_or_device}")
            
            source_bin = Gst.Bin.new(bin_name)
            uri_decode = self.make("uridecodebin", f"uri-decode-{index}")
            uri_decode.set_property("uri", uri_or_device)
            
            def on_pad_added(decodebin, pad, data):
                caps = pad.get_current_caps()
                if caps:
                    struct = caps.get_structure(0)
                    if struct.get_name().startswith("video"):
                        nvconv = source_bin.get_by_name(f"nvconv-{index}")
                        sink_pad = nvconv.get_static_pad("sink")
                        if not sink_pad.is_linked():
                            pad.link(sink_pad)
            
            uri_decode.connect("pad-added", on_pad_added, None)
            
            nvconv = self.make("nvvideoconvert", f"nvconv-{index}")
            caps = self.make("capsfilter", f"caps-{index}")
            caps.set_property("caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12"
            ))
            
            source_bin.add(uri_decode)
            source_bin.add(nvconv)
            source_bin.add(caps)
            nvconv.link(caps)
            
            # Ghost pad
            pad = caps.get_static_pad("src")
            ghost_pad = Gst.GhostPad.new("src", pad)
            source_bin.add_pad(ghost_pad)
        
        return source_bin
    
    def osd_probe(self, pad, info, udata):
        """Probe for FPS and text overlay"""
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            # FPS calculation
            sid = frame_meta.source_id
            now = time.monotonic()
            ts = _fps_ts[sid]
            ts.append(now)
            fps = 0.0
            if len(ts) >= 2:
                dt = ts[-1] - ts[0]
                if dt > 0:
                    fps = (len(ts) - 1) / dt
            
            # Text overlay
            disp = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            disp.num_labels = 1
            txt = disp.text_params[0]
            
            # Include source ID for multi-camera
            source_name = f"Cam{sid}" if self.args.multi else "Main"
            txt.display_text = f"{source_name} | Frame {frame_meta.frame_num} | Objects: {frame_meta.num_obj_meta} | FPS: {fps:.1f}"
            txt.x_offset = 20
            txt.y_offset = 30 + (sid * 40)  # Offset for multiple sources
            txt.font_params.font_name = "Sans"
            txt.font_params.font_size = 16
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.4)
            
            pyds.nvds_add_display_meta_to_frame(frame_meta, disp)
            
            # Update global metrics
            metrics["fps"] = fps
            metrics["frame"] = frame_meta.frame_num
            
            l_frame = l_frame.next
        
        return Gst.PadProbeReturn.OK
    
    def analytics_probe(self, pad, info, udata):
        """Metrics extraction and GPIO alerts"""
        global metrics
        
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            
            # Count objects and find largest
            counts = defaultdict(int)
            largest_area = 0
            largest_label = "none"
            
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                
                label = obj_meta.obj_label or "unknown"
                counts[label] += 1
                
                # Track largest (from group-mate's approach)
                if self.args.largest_only:
                    rect = obj_meta.rect_params
                    area = rect.width * rect.height
                    if area > largest_area:
                        largest_area = area
                        largest_label = label
                
                # GPIO alerts 
                if HAS_GPIO and obj_meta.confidence > 0.8:
                    if "cow" in label.lower():
                        GPIO.output(COW_GPIO_PIN, GPIO.HIGH)
                        time.sleep(0.1)
                        GPIO.output(COW_GPIO_PIN, GPIO.LOW)
                    elif "sheep" in label.lower():
                        GPIO.output(SHEEP_GPIO_PIN, GPIO.HIGH)
                        time.sleep(0.1)
                        GPIO.output(SHEEP_GPIO_PIN, GPIO.LOW)
                    elif "pig" in label.lower():
                        GPIO.output(PIG_GPIO_PIN, GPIO.HIGH)
                        time.sleep(0.1)
                        GPIO.output(PIG_GPIO_PIN, GPIO.LOW)
                
                l_obj = l_obj.next
            
            # Update metrics
            metrics.update({
                "total": sum(counts.values()),
                "cows": counts.get("cow", 0),
                "pigs": counts.get("pig", 0),
                "sheep": counts.get("sheep", 0),
                "largest_animal": largest_label,
                "counts_detail": dict(counts)
            })
            
            # Check for alerts
            alerts = []
            if metrics["total"] > 20:
                alerts.append(f"High density: {metrics['total']} animals")
            if metrics["total"] == 0 and frame_meta.frame_num > 100:
                alerts.append("No animals detected")
            
            metrics["alerts"] = alerts
            
            # Save to file for web dashboard
            with open('metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Print periodic updates
            if frame_meta.frame_num % 30 == 0:
                print(f"Frame {frame_meta.frame_num}: {metrics['total']} animals "
                      f"(Cows: {metrics['cows']}, Pigs: {metrics['pigs']}, "
                      f"Sheep: {metrics['sheep']})")
            
            l_frame = l_frame.next
        
        return Gst.PadProbeReturn.OK
    
    def signal_handler(self, sig_num, user_data=None):
        """Graceful shutdown handler"""
        global shutting_down
        
        if not shutting_down["flag"]:
            shutting_down["flag"] = True
            print(f"\nSignal {sig_num} received, sending EOS...")
            
            try:
                self.pipeline.send_event(Gst.Event.new_eos())
            except:
                self.loop.quit()
            
            # Force quit after timeout
            def force_quit():
                if shutting_down["flag"]:
                    print("EOS timeout, forcing shutdown...")
                    self.loop.quit()
                return False
            
            GLib.timeout_add_seconds(5, force_quit)
        else:
            print("Second signal, forcing immediate shutdown...")
            self.loop.quit()
        
        return True
    
    def build_pipeline(self):
        """Build the complete pipeline"""
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("livestock-pipeline")
        
        # Create sources
        sources = []
        if self.args.csi:
            sources.append(("csi", 0))
        if self.args.ip:
            for i, ip in enumerate(self.args.ip):
                url = f"http://{ip}:8080/video"
                sources.append((url, i + (1 if self.args.csi else 0)))
        
        if not sources:
            # Default to one IP camera
            sources = [("http://192.168.1.100:8080/video", 0)]
        
        # Streammux
        streammux = self.make("nvstreammux", "mux")
        streammux.set_property("batch-size", len(sources))
        streammux.set_property("width", 640)
        streammux.set_property("height", 640)
        streammux.set_property("batched-push-timeout", 33000)
        streammux.set_property("live-source", 1)
        self.pipeline.add(streammux)
        
        # Add sources and link to mux
        for i, (source, idx) in enumerate(sources):
            source_bin = self.create_source_bin(idx, source)
            self.pipeline.add(source_bin)
            
            # Queue with leak prevention 
            queue = self.make("queue", f"queue-{i}")
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 4)
            self.pipeline.add(queue)
            
            # Link source to queue to mux
            source_bin.link(queue)
            
            srcpad = queue.get_static_pad("src")
            sinkpad = streammux.get_request_pad(f"sink_{i}")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source {i} to mux")
        
        # Inference
        pgie = self.make("nvinfer", "pgie")
        pgie.set_property("config-file-path", self.args.pgie_config)

        # Object Tracker
        tracker = self.make("nvtracker", "tracker")
        tracker.set_property("display-tracking-id", 1)
        
        if self.args.tracker == "nvdcf":
            # Use NvDCF tracker 
            tracker.set_property("ll-lib-file", 
                "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            tracker.set_property("ll-config-file",
                "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
            tracker.set_property("compute-hw", 1)  # GPU
        else:
            # Use IOU tracker
            tracker.set_property("ll-lib-file",
                "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            tracker.set_property("ll-config-file",
                "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml")
        
        # Convert to RGBA for OSD
        nvvidconv1 = self.make("nvvideoconvert", "conv1")
        filter1 = self.make("capsfilter", "filter1")
        filter1.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=RGBA"
        ))
        
        # OSD
        nvosd = self.make("nvdsosd", "osd")
        nvosd.set_property("process-mode", 1)  # GPU mode
        
        # Tee for multiple outputs
        tee = self.make("tee", "tee")
        
        # Branch 1: Display
        queue_display = self.make("queue", "queue-display")
        
        if self.args.headless:
            sink = self.make("fakesink", "sink")
        else:
            # Convert for display
            nvvidconv2 = self.make("nvvideoconvert", "conv2")
            filter2 = self.make("capsfilter", "filter2")
            filter2.set_property("caps", Gst.Caps.from_string(
                "video/x-raw,format=RGBA"
            ))
            videoconv = self.make("videoconvert", "vconv")
            sink = self.make("ximagesink", "sink")
            sink.set_property("sync", 0)
        
        # Branch 2: RTSP output
        if self.args.rtsp:
            queue_rtsp = self.make("queue", "queue-rtsp")
            nvvidconv_rtsp = self.make("nvvideoconvert", "conv-rtsp")
            filter_rtsp = self.make("capsfilter", "filter-rtsp")
            filter_rtsp.set_property("caps", Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=I420"
            ))
            encoder = self.make("nvv4l2h264enc", "encoder")
            encoder.set_property("bitrate", 4000000)
            rtppay = self.make("rtph264pay", "rtppay")
            
            udpsink = self.make("udpsink", "udpsink")
            udpsink.set_property("host", "224.1.1.1")
            udpsink.set_property("port", 5000)
            udpsink.set_property("async", False)
            udpsink.set_property("sync", 1)
        
        # Add all elements to pipeline
        elements = [pgie, tracker, nvvidconv1, filter1, nvosd, tee, queue_display]
        
        if not self.args.headless:
            elements.extend([nvvidconv2, filter2, videoconv])
        elements.append(sink)
        
        if self.args.rtsp:
            elements.extend([queue_rtsp, nvvidconv_rtsp, filter_rtsp, 
                           encoder, rtppay, udpsink])
        
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link main pipeline
        self.link_many(streammux, pgie, tracker, nvvidconv1, filter1, nvosd, tee)
        
        # Link display branch
        tee.link(queue_display)
        if self.args.headless:
            queue_display.link(sink)
        else:
            self.link_many(queue_display, nvvidconv2, filter2, videoconv, sink)
        
        # Link RTSP branch
        if self.args.rtsp:
            tee.link(queue_rtsp)
            self.link_many(queue_rtsp, nvvidconv_rtsp, filter_rtsp, 
                          encoder, rtppay, udpsink)
        
        # Add probes
        osd_sink_pad = nvosd.get_static_pad("sink")
        probe_id = osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, 
                                          self.osd_probe, None)
        self.probe_ids.append((osd_sink_pad, probe_id))
        
        analytics_pad = tracker.get_static_pad("src")
        probe_id = analytics_pad.add_probe(Gst.PadProbeType.BUFFER,
                                           self.analytics_probe, None)
        self.probe_ids.append((analytics_pad, probe_id))
    
    def run(self):
        """Run the pipeline"""
        self.build_pipeline()
        
        # Setup event handling
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_callback, self.loop)
        
        # Register signal handlers
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, 
                            self.signal_handler, None)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM,
                            self.signal_handler, None)
        
        print("Starting pipeline...")
        print(f"Mode: {'Headless' if self.args.headless else 'Display'}")
        print(f"Tracker: {self.args.tracker}")
        print(f"RTSP output: {'Enabled on udp://224.1.1.1:5000' if self.args.rtsp else 'Disabled'}")
        
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            print("Cleaning up...")
            
            # Remove probes
            for pad, probe_id in self.probe_ids:
                try:
                    pad.remove_probe(probe_id)
                except:
                    pass
            
            # Stop pipeline
            self.pipeline.set_state(Gst.State.NULL)
            
            # Cleanup GPIO
            if HAS_GPIO:
                GPIO.cleanup()
            
            print("Shutdown complete")

def run_webserver():
    """Web server for dashboard (your approach)"""
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  
    
    httpd = HTTPServer(('0.0.0.0', 8000), QuietHandler)
    print("📊 Dashboard available at: http://localhost:8000/metrics.html")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(
        description="Production Livestock Monitoring"
    )
    
    # Camera sources
    parser.add_argument("--csi", action="store_true") # Use CSI camera
    parser.add_argument("--ip", nargs="+") # IP addresses of phones 
    parser.add_argument("--multi", action="store_true") # Multi-camera mode
    
    # Model config
    parser.add_argument("--pgie-config", default="config_infer.txt",
                       help="Primary inference config file")
    
    # Tracker selection
    parser.add_argument("--tracker", choices=["iou", "nvdcf"], default="iou",
                       help="Tracker type (iou=faster, nvdcf=accurate)")
    
    # Processing options
    parser.add_argument("--largest-only", action="store_true",
                       help="Track only largest animal")
    parser.add_argument("--headless", action="store_true",
                       help="Run without display")
    parser.add_argument("--rtsp", action="store_true",
                       help="Enable RTSP streaming output")
    
    # Web dashboard
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Disable web dashboard")
    
    args = parser.parse_args()
    
    # Start web server if enabled
    if not args.no_dashboard:
        web_thread = threading.Thread(target=run_webserver, daemon=True)
        web_thread.start()
    
    # Run pipeline
    monitor = LivestockMonitor(args)
    monitor.run()

if __name__ == "__main__":
    main()