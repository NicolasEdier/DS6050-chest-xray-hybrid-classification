"""
Verification script to ensure everything is ready for hybrid model training
Run this before starting the full training pipeline
"""

import sys
from pathlib import Path
import torch


def check_file(path, description):
    """Check if a file exists"""
    if Path(path).exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False


def check_directory(path, description):
    """Check if directory exists and has content"""
    path = Path(path)
    if path.exists():
        count = len(list(path.rglob('*')))
        print(f"  ✓ {description} ({count} items)")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False


def check_import(module_name, package=None):
    """Check if a module can be imported"""
    try:
        if package:
            __import__(package)
        else:
            __import__(module_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError:
        print(f"  ✗ {module_name} - MISSING")
        return False


def main():
    print("\n" + "="*80)
    print("HYBRID MODEL TRAINING - SETUP VERIFICATION")
    print("="*80)
    
    all_checks = []
    
    # ========================================================================
    # 1. Check data preprocessing
    # ========================================================================
    print("\n1. Data Preprocessing")
    print("-" * 80)
    
    checks = [
        check_file('processed_data/train.csv', 'Training data CSV'),
        check_file('processed_data/val.csv', 'Validation data CSV'),
        check_file('processed_data/test.csv', 'Test data CSV'),
        check_file('processed_data/class_weights.json', 'Class weights'),
        check_directory('NIH_ChestXray', 'Image directory')
    ]
    all_checks.extend(checks)
    
    # ========================================================================
    # 2. Check model files
    # ========================================================================
    print("\n2. Model Implementation")
    print("-" * 80)
    
    checks = [
        check_file('models/baseline.py', 'Baseline models'),
        check_file('models/hybrid.py', 'Hybrid models (NEW)'),
    ]
    all_checks.extend(checks)
    
    # ========================================================================
    # 3. Check training files
    # ========================================================================
    print("\n3. Training Scripts")
    print("-" * 80)
    
    checks = [
        check_file('training/train.py', 'Baseline training'),
        check_file('training/train_hybrid.py', 'Hybrid training (NEW)'),
        check_file('training/losses.py', 'Loss functions'),
        check_file('training/metrics.py', 'Metrics'),
    ]
    all_checks.extend(checks)
    
    # ========================================================================
    # 4. Check evaluation files
    # ========================================================================
    print("\n4. Evaluation Scripts")
    print("-" * 80)
    
    checks = [
        check_file('evaluation/evaluate.py', 'Evaluation script'),
        check_file('evaluation/visualize.py', 'Visualization script'),
        check_file('evaluation/compare_models.py', 'Model comparison'),
    ]
    all_checks.extend(checks)
    
    # ========================================================================
    # 5. Check dependencies
    # ========================================================================
    print("\n5. Python Dependencies")
    print("-" * 80)
    
    checks = [
        check_import('torch'),
        check_import('torchvision'),
        check_import('timm'),
        check_import('numpy'),
        check_import('pandas'),
        check_import('sklearn', 'scikit-learn'),
        check_import('PIL', 'pillow'),
        check_import('cv2', 'opencv-python'),
        check_import('tqdm'),
        check_import('matplotlib'),
        check_import('seaborn'),
    ]
    all_checks.extend(checks)
    
    # ========================================================================
    # 6. Check GPU
    # ========================================================================
    print("\n6. GPU Availability")
    print("-" * 80)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU available: {device_name}")
        print(f"  ✓ GPU memory: {memory_gb:.1f} GB")
        
        if memory_gb < 10:
            print("  ⚠ Warning: GPU has less than 10GB memory")
            print("    Recommendation: Use batch_size=8 instead of 16")
        
        all_checks.append(True)
    else:
        print("  ✗ No GPU available")
        print("  ⚠ Training will be VERY slow on CPU")
        all_checks.append(False)
    
    # ========================================================================
    # 7. Check baseline results
    # ========================================================================
    print("\n7. Baseline Results")
    print("-" * 80)
    
    baseline_checkpoints = [
        ('checkpoints/resnet50/best_model.pth', 'ResNet-50'),
        ('checkpoints/densenet121/best_model.pth', 'DenseNet-121'),
        ('checkpoints/resnet50_multiscale/best_model.pth', 'ResNet-50 Multiscale'),
    ]
    
    baseline_exists = False
    for checkpoint, name in baseline_checkpoints:
        if check_file(checkpoint, f'{name} checkpoint'):
            baseline_exists = True
    
    if not baseline_exists:
        print("\n  ⚠ No baseline checkpoints found")
        print("    You may want to train baselines first for comparison")
    
    # ========================================================================
    # 8. Test hybrid model creation
    # ========================================================================
    print("\n8. Test Hybrid Model Creation")
    print("-" * 80)
    
    try:
        from models.hybrid import get_hybrid_model
        
        for variant in ['A0', 'A3', 'A4', 'A5']:
            model = get_hybrid_model(variant=variant, num_classes=14, pretrained=False)
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ {variant}: {params:,} parameters")
            del model  # Free memory
        
        all_checks.append(True)
        
    except Exception as e:
        print(f"  ✗ Error creating models: {e}")
        all_checks.append(False)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(all_checks)
    total = len(all_checks)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nYou are ready to train hybrid models:")
        print("  python run_hybrid_experiments.py")
        print("\nOr train individual variants:")
        print("  python training/train_hybrid.py --variant A0 --epochs 25")
        
    else:
        print(f"\n✗ {total - passed} CHECKS FAILED")
        print("\nPlease fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing data: python data/preprocess.py")
        print("  - Missing model files: Copy hybrid.py to models/")
        print("  - Missing training script: Copy train_hybrid.py to training/")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)