'''
    2025.03.18 data 6th version
'''

from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class IRDropConfig:
    mean: float
    std: float
    min: float
    max: float

@dataclass(frozen=True)
class Config:
    resolution: tuple = (256, 256)
    interpolation: int = cv2.INTER_AREA
    ir_drop: IRDropConfig = None


ir_drop_200nm = IRDropConfig(
    mean=2.5323e-03,
    std=2.0482e-03,
    min=0.0,
    max=4.7579e-02
)

ir_drop_100nm = IRDropConfig(
    mean=1.8642e-03,
    std=2.2837e-03,
    min=0.0,
    max=4.7579e-02
)

ir_drop_500nm = IRDropConfig(
    mean=2.7501e-03,
    std=2.1259e-03,
    min=0.0,
    max=4.7579e-02
)

ir_drop_1um = IRDropConfig(
    # mean=4.8804e-02,
    # std=4.6042e-02,
    mean=2.9581e-03,
    std=2.3594e-03,
    min=0.0,
    max=4.7579e-02
)


def get_config(process_node):
    """
    공정 노드에 따른 Config 객체를 반환합니다.
    
    Args:
        process_node: 공정 노드 ("100nm", "200nm", "500nm", "1um" 중 하나)
        
    Returns:
        Config 객체
    """
    if process_node == "100nm":
        ir_drop = ir_drop_100nm
    elif process_node == "200nm":
        ir_drop = ir_drop_200nm
    elif process_node == "500nm":
        ir_drop = ir_drop_500nm
    elif process_node == "1um":
        ir_drop = ir_drop_1um
    else:
        raise ValueError(f"지원하지 않는 공정 노드입니다: {process_node}")
    
    return Config(ir_drop=ir_drop)


# 출력 예시
if __name__ == "__main__":
    for node in ["100nm", "200nm", "500nm", "1um"]:
        config = get_config(node)
        print(f"{node} IR Drop 통계:")
        print(f"  평균: {config.ir_drop.mean:.8f}")
        print(f"  표준편차: {config.ir_drop.std:.8f}")
        print(f"  최소값: {config.ir_drop.min:.8f}")
        print(f"  최대값: {config.ir_drop.max:.8f}")
        print()