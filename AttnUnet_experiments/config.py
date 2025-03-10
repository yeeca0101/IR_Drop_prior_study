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
    # 네트워크 입력 해상도
    resolution: tuple = (256, 256)
    # cv2.resize 보간법 (예: cv2.INTER_LANCZOS4)
    interpolation: int = cv2.INTER_NEAREST
    # IR Drop 관련 통계정보
    ir_drop: IRDropConfig = None


# 200nm 기준값 (원본)
ir_drop_200nm = IRDropConfig(
    mean=0.00140431,
    std=0.00247082,
    min=0.0,
    max=0.0531378999999999
)

# 100nm 값 (200nm의 0.5배)
ir_drop_100nm = IRDropConfig(
    mean=-1,
    std=-1,
    min=0.0,
    max=0.0904707999999998
)

# 500nm 값 (200nm의 2.5배)
ir_drop_500nm = IRDropConfig(
    mean=-1,
    std=-1,
    min=0.0,
    max=0.0353812279999999
)

# 1um (1000nm) 값 (200nm의 5배)
ir_drop_1um = IRDropConfig(
    mean=-1,
    std=-1,
    min=0.0,
    max=0.0250380959999999
)


# 사용 예시
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