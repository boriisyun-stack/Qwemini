import pygame
import sys
import math

# 초기화
pygame.init()

# 화면 설정
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("이중진자 시뮬레이션")
clock = pygame.time.Clock()

# 상수
G = 9.81 # 중력 가속도 (m/s²)
L1 = 150 # 첫 번째 막대 길이
L2 = 150 # 두 번째 막대 길이
M1 = 10 # 첫 번째 질량
M2 = 10 # 두 번째 질량
DT = 0.05 # 시간 간격 (초)

# 초기 각도 및 각속도 (라디안)
theta1 = math.pi / 2 # 90도
theta2 = math.pi / 2
omega1 = 0.0
omega2 = 0.0

# 궤적 저장
trace = []

# 색상
BG_COLOR = (10, 10, 30)
PENDULUM_COLOR = (200, 200, 255)
TRACE_COLOR = (100, 150, 255, 50)

# 이중진자 운동 방정식 (라그랑지안 기반)
def derivatives(theta1, theta2, omega1, omega2):
    num1 = -G * (2 * M1 + M2) * math.sin(theta1)
    num2 = -M2 * G * math.sin(theta1 - 2 * theta2)
    num3 = -2 * math.sin(theta1 - theta2) * M2 * (omega2**2 * L2 + omega1**2 * L1 * math.cos(theta1 - theta2))
    den = L1 * (2 * M1 + M2 - M2 * math.cos(2 * theta1 - 2 * theta2))
    alpha1 = (num1 + num2 + num3) / den

    num1 = 2 * math.sin(theta1 - theta2)
    num2 = (omega1**2 * L1 * (M1 + M2)) + G * (M1 + M2) * math.cos(theta1) + omega2**2 * L2 * M2 * math.cos(theta1 - theta2)
    den = L2 * (2 * M1 + M2 - M2 * math.cos(2 * theta1 - 2 * theta2))
    alpha2 = (num1 * num2) / den

    return omega1, omega2, alpha1, alpha2

# 시뮬레이션 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 운동 방정식 업데이트
    dtheta1, dtheta2, domega1, domega2 = derivatives(theta1, theta2, omega1, omega2)
    theta1 += dtheta1 * DT
    theta2 += dtheta2 * DT
    omega1 += domega1 * DT
    omega2 += domega2 * DT

    # 위치 계산 (중심: 화면 상단 중앙)
    x1 = WIDTH // 2 + L1 * math.sin(theta1)
    y1 = HEIGHT // 4 + L1 * math.cos(theta1)
    x2 = x1 + L2 * math.sin(theta2)
    y2 = y1 + L2 * math.cos(theta2)

    # 궤적 추가 (최대 200점)
    trace.append((int(x2), int(y2)))
    if len(trace) > 200:
        trace.pop(0)

    # 화면 클리어
    screen.fill(BG_COLOR)

    # 궤적 그리기
    if len(trace) > 1:
        for i in range(1, len(trace)):
            alpha = int(255 * i / len(trace))
            color = (*TRACE_COLOR[:3], alpha)
            pygame.draw.line(screen, color, trace[i-1], trace[i], 1)

    # 진자 막대 그리기
    pygame.draw.line(screen, PENDULUM_COLOR, (WIDTH//2, HEIGHT//4), (int(x1), int(y1)), 3)
    pygame.draw.line(screen, PENDULUM_COLOR, (int(x1), int(y1)), (int(x2), int(y2)), 3)

    # 질량 원 그리기
    pygame.draw.circle(screen, PENDULUM_COLOR, (int(x1), int(y1)), M1)
    pygame.draw.circle(screen, PENDULUM_COLOR, (int(x2), int(y2)), M2)

    # 화면 업데이트
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
