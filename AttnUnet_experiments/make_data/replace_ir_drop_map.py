import os
import shutil

def validate_and_replace_ir_drops():
    """
    1. Copies *_ir_drop.npy files from pdn_data_5th to target directories
    2. Validates corresponding {index}_power.npy existence in target
    3. Removes mismatched ir_drop files lacking power counterparts
    """
    # Configuration
    data_root = '/data'
    source_root = "pdn_data_5th"
    target_roots = ["pdn_3rd_4types", "pdn_4th_4types"]
    size_groups = ["1um_numpy", "100nm_numpy", "200nm_numpy", "500nm_numpy"]

    # Statistics tracking
    copied_count = 0
    deleted_count = 0
    error_count = 0

    for target_root in target_roots:
        for size_dir in size_groups:
            # Build directory paths
            src_dir = os.path.join(data_root,source_root, size_dir)
            tgt_dir = os.path.join(data_root,target_root, size_dir)

            # Validate directory structure
            if not os.path.exists(src_dir):
                print(f"🔴 Source missing: {src_dir}")
                error_count += 1
                continue
                
            if not os.path.exists(tgt_dir):
                print(f"🔴 Target missing: {tgt_dir}")
                error_count += 1
                continue

            # Process files
            try:
                ir_files = [f for f in os.listdir(src_dir) 
                           if f.endswith('_ir_drop.npy')]
            except Exception as e:
                print(f"🔴 Listing failed: {src_dir} - {str(e)}")
                error_count += 1
                continue

            for ir_file in ir_files:
                src_path = os.path.join(src_dir, ir_file)
                tgt_path = os.path.join(tgt_dir, ir_file)

                # File copy operation
                try:
                    shutil.copy2(src_path, tgt_path)
                    copied_count += 1
                    print(f"🟢 Copied: {tgt_path}")
                except Exception as e:
                    print(f"🔴 Copy failed: {src_path} → {tgt_path} - {str(e)}")
                    error_count += 1
                    continue

                # Power file validation
                try:
                    base_name = os.path.splitext(ir_file)[0]
                    if not base_name.endswith('_ir_drop'):
                        print(f"🟠 Bad format: {ir_file}")
                        error_count += 1
                        continue

                    index = base_name.rsplit('_ir_drop', 1)[0]
                    power_file = f"{index}_current.npy"
                    power_path = os.path.join(tgt_dir, power_file)

                    if not os.path.exists(power_path):
                        os.remove(tgt_path)
                        deleted_count += 1
                        print(f"🟡 Deleted: {tgt_path} (Missing {power_file})")
                except Exception as e:
                    print(f"🔴 Validation failed: {tgt_path} - {str(e)}")
                    error_count += 1

    # Final report
    print(f"\n📊 Final Results:")
    print(f"✅ Successfully copied: {copied_count}")
    print(f"🗑️  Removed mismatches: {deleted_count}")
    print(f"❌ Total errors: {error_count}")
    print(f"🔢 Net changes: {copied_count - deleted_count} files updated")


def replace_ir_drop_files():
    """
    조건부 파일 교체 시스템
    1. pdn_data_5th의 *_ir_drop.npy 파일을 
    2. 대상 디렉토리(pdn_3rd_4types, pdn_4th_4types)에 
    3. 동일 인덱스의 _power.npy가 존재할 때만 복사
    """
    data_root = '/data'
    source_base = os.path.join(data_root, "pdn_data_5th")
    target_parents = ["pdn_3rd_4types", "pdn_4th_4types"]
    size_groups = ["1um_numpy", "100nm_numpy", "200nm_numpy", "500nm_numpy"]

    stats = {'copied': 0, 'skipped': 0, 'errors': 0}

    for target_parent in target_parents:
        for size_dir in size_groups:
            src_dir = os.path.join(source_base, size_dir)
            tgt_dir = os.path.join(data_root, target_parent, size_dir)

            # 디렉토리 유효성 검증
            if not all(map(os.path.exists, [src_dir, tgt_dir])):
                stats['errors'] += 1
                continue

            # 파일 처리 파이프라인
            try:
                for fname in os.listdir(src_dir):
                    if not fname.endswith('_ir_drop.npy'):
                        continue
                    
                    # 파워 파일 매칭 검증
                    base_id = fname.rsplit('_ir_drop.npy', 1)[0]
                    power_file = f"{base_id}_current.npy"
                    power_path = os.path.join(tgt_dir, power_file)
                    
                    if not os.path.exists(power_path):
                        stats['skipped'] += 1
                        continue
                        
                    # 실제 복사 작업
                    src_path = os.path.join(src_dir, fname)
                    dst_path = os.path.join(tgt_dir, fname)
                    shutil.copy2(src_path, dst_path)
                    stats['copied'] += 1
                    print(f"Copied: {dst_path}")
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"Error: {str(e)}")

    # 최종 리포트
    print(f"\nResults:")
    print(f"✅ Copied: {stats['copied']}")
    print(f"⚠️ Skipped: {stats['skipped']} (no power file)")
    print(f"🚨 Errors: {stats['errors']}")

replace_ir_drop_files()
# validate_and_replace_ir_drops()
