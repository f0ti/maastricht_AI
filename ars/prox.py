import pygame
import numpy as np
from pygame.locals import *
from math import hypot as hyp

pygame.init()

screen = pygame.display.set_mode((600,500))
clock = pygame.time.Clock()

colour = (0, 255, 0)

# player

rect = pygame.Rect(0, 0, 20, 20)
rect.center = screen.get_rect().center
vel = 5

# generate random obstacles coordinates

NR_OBSTACLES = 5

# np.reshape([np.random.randint(1, 100, 5), np.random.randint(1, 100, 5)], (5, 2))

obstacles = np.array((np.random.randint(1, 600, NR_OBSTACLES), np.random.randint(1, 500, NR_OBSTACLES))).T

while 1:
    clock.tick(60)

    center = (300, 250)
    radius = 100
    
    # particle

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            print(pygame.key.name(event.key))

    keys = pygame.key.get_pressed()
    
    rect.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * vel
    rect.y += (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * vel
    
    rect.centerx = rect.centerx % screen.get_width()
    rect.centery = rect.centery % screen.get_height()

    screen.fill(0)
    pygame.draw.rect(screen, (255, 0, 0), rect)

    # draw obstacles
    for obs_pos in obstacles:
        obstacle = pygame.Rect(obs_pos, (20, 20))
        pygame.draw.rect(screen, (255, 255, 255), obstacle)

    pygame.display.flip()
