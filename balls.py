"""
Some kind of metaballs thing. One ball is stationary, and the other's
location follows the mouse.

Source and inspiration: Tsoding Daily's youtube video
https://www.youtube.com/watch?v=PUu5kjoWw0k
"""

import cv2
import json
import math
import numba as nb
import numpy as np
import pyautogui as pag


def read_config() -> dict:
    """
    Reads the config file. Converts the lists of bgr colors to tuples.
    :return:
    """
    with open("config.json", "r") as f:
        data = json.load(f)
    for k, v in data.items():
        if isinstance(v, list) and "COLOR" in k:
            data[k] = tuple([x / 255 for x in v])

    return data


def _scale(a: float, b: float, p: float) -> float:
    """
    :param a:
    :param b:
    :param p:
    :return:
    """
    return p * a + (1 - p) * b


def _blend_pixels(ball1_color: tuple, ball2_color: tuple, p: float) -> tuple:
    """
    Blends the colors of the balls according to the given factor
    :param ball1_color:
    :param ball2_color:
    :param p:
    :return:
    """
    b = _scale(ball1_color[0], ball2_color[0], p)
    g = _scale(ball1_color[1], ball2_color[1], p)
    r = _scale(ball1_color[2], ball2_color[2], p)
    return b, g, r


@nb.jit(nopython=True)
def _color_pixels(img: np.ndarray, ball1_pos: np.ndarray, ball2_pos: np.ndarray,
                  tol: float, ball1_color: tuple, ball2_color: tuple,
                  bg_color: tuple) -> np.ndarray:
    """
    Colors the pixels
    :param img: The image
    :param ball1_pos:
    :param ball2_pos: The updated position of the moving ball
    :param tol:
    :param ball1_color:
    :param ball2_color:
    :param bg_color:
    :return:
    """
    b1_x, b1_y = ball1_pos
    b2_x, b2_y = ball2_pos
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            xp = x + 0.5
            yp = y + 0.5
            dx1, dy1 = xp - b1_x, yp - b1_y
            dx2, dy2 = xp - b2_x, yp - b2_y
            d1 = 1 / math.sqrt(dx1 * dx1 + dy1 * dy1)
            d2 = 1 / math.sqrt(dx2 * dx2 + dy2 * dy2)
            d = d1 + d2
            if d > tol:
                p = d1 / d
                r1, g1, b1 = ball1_color
                r2, g2, b2 = ball2_color
                r = p * r1 + (1 - p) * r2
                g = p * g1 + (1 - p) * g2
                b = p * b1 + (1 - p) * b2
                img[y, x] = (r, g, b)
            else:
                img[y, x] = bg_color

    return img


def _update_img(img: np.ndarray, ball1_pos: np.ndarray, ball1_color: tuple,
                ball2_color: tuple, bg_color: tuple, tol: int | float,
                pad_x: int, pad_y: int) -> np.ndarray:
    """
    :param img:
    :param ball1_pos: The position of the stationary ball
    :param ball1_color: Color of the stationary ball
    :param ball2_color: Color of the moving ball
    :param bg_color: Background color
    :param tol: The tolerance
    :param pad_x:
    :param pad_y:
    :return:
    """
    pos = pag.position()
    ball2_pos = np.array([pos.x - pad_x, pos.y - pad_y])
    img = _color_pixels(img, ball1_pos, ball2_pos, tol, ball1_color,
                        ball2_color, bg_color)
    return img


def simulate(config: dict) -> None:
    """
    Metaballs simulation with a single thread
    :param config:
    :return:
    """
    w = config["WIDTH"]
    h = config["HEIGHT"]
    ball1_pos = np.array(config["BALL1_POS"])
    ball1_color = config["BALL1_COLOR"]
    ball2_color = config["BALL2_COLOR"]
    bg_color = config["BG_COLOR"]
    tol = config["TOLERANCE"]
    fps = config["FPS"]
    img = np.zeros((h, w, 3))
    pad_x, pad_y = 100, 100
    name = "Metaballs"
    cv2.namedWindow(name)
    while True:
        img = _update_img(img, ball1_pos, ball1_color, ball2_color,
                          bg_color, tol, pad_x, pad_y)
        cv2.moveWindow(name, pad_x, pad_y)
        cv2.imshow(name, img)
        cv2.waitKey(int(1 / fps * 1000))


def main():
    config = read_config()
    simulate(config)


if __name__ == "__main__":
    main()
