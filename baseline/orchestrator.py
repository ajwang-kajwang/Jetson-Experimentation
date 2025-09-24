# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Baseline PyTorch Inference for Lab 5 Task 0'
    )
    parser.add_argument(
        '--mode', 
        choices=['live', 'coco', 'both'], 
        default='live',
        help='Inference mode'
    )
    parser.add_argument(
        '--camera', 
        type=int, 
        default=Config.CAMERA_INDEX,
        help='Camera device index'
    )
    parser.add_argument(
        '--metrics_out', 
        type=str, 
        default=None,
        help='Output JSON file for metrics'
    )
    parser.add_argument(
        '--no-display', 
        action='store_true',
        help='Disable video display'
    )
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = BaselineInferenceEngine()
    
    # Run inference based on mode
    if args.mode in ['live', 'both']:
        output_file = args.metrics_out or 'baseline_live_metrics.json'
        metrics = engine.run_live_inference(
            camera_index=args.camera,
            display=not args.no_display
        )
        engine.save_metrics(metrics, output_file)
    
    if args.mode in ['coco', 'both']:
        output_file = args.metrics_out or 'baseline_coco_metrics.json'
        metrics = engine.run_coco_validation()
        engine.save_metrics(metrics, output_file)

if __name__ == '__main__':
    main()