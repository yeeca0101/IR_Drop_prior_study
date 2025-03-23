import json
from dataclasses import dataclass, field
import cv2
from datetime import datetime

# JSON 파일에 저장된 통계정보의 구조와 동일한 필드를 갖는 dataclass들
@dataclass(frozen=True)
class StatConfig:
    mean: float
    std: float
    min: float
    max: float

@dataclass(frozen=True)
class ResistanceStats:
    # 각 layer별 통계와 전체 통계를 포함합니다.
    layers: dict = field(default_factory=dict)  # 예: {"m8": StatConfig, "via56": StatConfig, ...}
    total: StatConfig = None

@dataclass(frozen=True)
class Config:
    resolution: tuple = (256, 256)
    interpolation: int = cv2.INTER_AREA
    ir_drop: StatConfig = None
    current: StatConfig = None
    pad: StatConfig = None
    resistances: ResistanceStats = None
    meta: dict = field(default_factory=dict)

def map_json_to_config(json_data):
    """
    저장된 JSON 데이터를 Config 객체로 매핑합니다.
    """
    stats = json_data.get("stats", {})
    meta = json_data.get("meta", {})

    def create_stat_config(stat_dict):
        if stat_dict is None or stat_dict.get("mean") is None:
            return None
        return StatConfig(
            mean=stat_dict["mean"],
            std=stat_dict["std"],
            min=stat_dict["min"],
            max=stat_dict["max"]
        )
    
    ir_drop_cfg = create_stat_config(stats.get("ir_drop"))
    current_cfg = create_stat_config(stats.get("current"))
    pad_cfg = create_stat_config(stats.get("pad"))
    
    resistance_stats = stats.get("resistance", {})
    layers_cfg = {}
    for layer_key, layer_val in resistance_stats.get("layers", {}).items():
        layers_cfg[layer_key] = create_stat_config(layer_val)
    total_cfg = create_stat_config(resistance_stats.get("total"))
    resistance_cfg = ResistanceStats(layers=layers_cfg, total=total_cfg)
    
    return Config(
        resolution=(256, 256),
        interpolation=cv2.INTER_AREA,
        ir_drop=ir_drop_cfg,
        current=current_cfg,
        pad=pad_cfg,
        resistances=resistance_cfg,
        meta=meta
    )

def get_config(a,config_json_path="/IR_Drop_prior_study/Prior_dev_lab/configs/stastics_1um.json"):
    """
    저장된 JSON 파일(config_json_path)을 읽어 Config 객체로 반환합니다.
    """
    with open(config_json_path, "r") as f:
        json_data = json.load(f)
    return map_json_to_config(json_data)

# 출력 예시
if __name__ == "__main__":
    config = get_config()
    
    print("IR Drop 통계:")
    if config.ir_drop:
        print(f"  평균: {config.ir_drop.mean:.8f}")
        print(f"  표준편차: {config.ir_drop.std:.8f}")
        print(f"  최소값: {config.ir_drop.min:.8f}")
        print(f"  최대값: {config.ir_drop.max:.8f}")
    else:
        print("  데이터 없음.")
    
    print("\nCurrent 통계:")
    if config.current:
        print(f"  평균: {config.current.mean:.8f}")
        print(f"  표준편차: {config.current.std:.8f}")
        print(f"  최소값: {config.current.min:.8f}")
        print(f"  최대값: {config.current.max:.8f}")
    else:
        print("  데이터 없음.")
    
    print("\nPad 통계:")
    if config.pad:
        print(f"  평균: {config.pad.mean:.8f}")
        print(f"  표준편차: {config.pad.std:.8f}")
        print(f"  최소값: {config.pad.min:.8f}")
        print(f"  최대값: {config.pad.max:.8f}")
    else:
        print("  데이터 없음.")
    
    print("\nResistance 전체 통계:")
    if config.resistances and config.resistances.total:
        print(f"  평균: {config.resistances.total.mean:.8f}")
        print(f"  표준편차: {config.resistances.total.std:.8f}")
        print(f"  최소값: {config.resistances.total.min:.8f}")
        print(f"  최대값: {config.resistances.total.max:.8f}")
    else:
        print("  데이터 없음.")
    
    print("\nResistance Layer별 통계:")
    if config.resistances and config.resistances.layers:
        for layer, stat in config.resistances.layers.items():
            print(f"Layer {layer}:")
            print(f"  평균: {stat.mean:.8f}")
            print(f"  표준편차: {stat.std:.8f}")
            print(f"  최소값: {stat.min:.8f}")
            print(f"  최대값: {stat.max:.8f}")
    else:
        print("  데이터 없음.")
