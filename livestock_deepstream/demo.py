#!/usr/bin/env python3
"""
Complete livestock monitoring with metrics
"""
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import argparse

# Global metrics
metrics = {"total": 0, "cows": 0, "pigs": 0, "sheep": 0, "fps": 0}
frame_count = 0

def probe_callback(pad, info, u_data):
    """Extract metrics from DeepStream pipeline"""
    global metrics, frame_count
    
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        frame_count += 1
        
        # Count objects by class
        counts = {}
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            label = obj_meta.obj_label
            counts[label] = counts.get(label, 0) + 1
            l_obj = l_obj.next
        
        # Update metrics
        metrics = {
            "total": sum(counts.values()),
            "cows": counts.get("cow", 0),
            "pigs": counts.get("pig", 0),
            "sheep": counts.get("sheep", 0),
            "persons": counts.get("person", 0),
            "frame": frame_count,
            "fps": frame_meta.fps if hasattr(frame_meta, 'fps') else 15
        }
        
        # Print every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {metrics['total']} animals detected")
        
        # Save metrics for web dashboard
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        l_frame = l_frame.next
    
    return Gst.PadProbeReturn.OK

def run_webserver():
    """Serve Dashboard"""
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  
    
    httpd = HTTPServer(('0.0.0.0', 8000), Handler)
    print("📊 Dashboard: http://localhost:8000/metrics.html")
    httpd.serve_forever()

def main(camera_ip):
    # Start web server
    threading.Thread(target=run_webserver, daemon=True).start()
    
    # Build pipeline
    pipeline_str = f"""
        uridecodebin uri=http://{camera_ip}:8080/video ! 
        nvstreammux name=mux batch-size=1 width=640 height=640 ! 
        nvinfer config-file-path=config_infer.txt ! 
        nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so 
                  ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml !
        nvvideoconvert ! 
        nvdsosd name=osd ! 
        nvegltransform ! 
        nveglglessink sync=0
    """
    
    print(f"🎥 Starting pipeline with camera: {camera_ip}")
    
    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Add probe for metrics
    osd = pipeline.get_by_name("osd")
    if osd:
        pad = osd.get_static_pad("sink")
        pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, 0)
    
    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    print("✅ Pipeline running. Press Ctrl+C to stop")
    
    # Run main loop
    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='192.168.1.100', 
                       help='IP address of phone running IP Webcam')
    args = parser.parse_args()
    
    main(args.ip)