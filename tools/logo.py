import cv2
import numpy as np

def main():
    logo = np.zeros((74, 224), dtype=np.uint8)
    title = "COMPUTER VISION TOOLKIT"
    capabilities = "CCs: 6.1,7.5,8.6"
    author = "Author: Zhiwei Zeng"
    version = "Version: 1.0"

    color = 255
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = .5
    thickness = 1
    textSize1, baseLine1 = cv2.getTextSize(title, fontFace, fontScale, thickness)
    x = (logo.shape[1] - textSize1[0]) // 2
    y = textSize1[1] + baseLine1
    org = (x, y)
    cv2.putText(logo, title, org, fontFace, fontScale, color, thickness)

    fontScale = .5
    textSize2, baseLine2 = cv2.getTextSize(capabilities, fontFace, fontScale, thickness)
    x = (logo.shape[1] - textSize2[0]) // 2
    y += (textSize2[1] + baseLine2)
    org = (x, y)
    cv2.putText(logo, capabilities, org, fontFace, fontScale, color, thickness)

    textSize3, baseLine3 = cv2.getTextSize(author, fontFace, fontScale, thickness)
    x = (logo.shape[1] - textSize3[0]) // 2
    y += (textSize3[1] + baseLine3)
    org = (x, y)
    cv2.putText(logo, author, org, fontFace, fontScale, color, thickness)

    textSize4, baseLine4 = cv2.getTextSize(version, fontFace, fontScale, thickness)
    x = (logo.shape[1] - textSize4[0]) // 2
    y += (textSize4[1] + baseLine4)
    org = (x, y)
    cv2.putText(logo, version, org, fontFace, fontScale, color, thickness)

    cv2.imwrite("logo.png", logo)

    with open('logo.h', 'w') as file:
        file.write('const char logo[] = {\n')
        for y in range(logo.shape[0]):
            for x in range(logo.shape[1]):
                if logo[y,x] > 0:
                    file.write('0x2A')
                else:
                    file.write('0x20')
                if y < logo.shape[0] - 1:
                    file.write(',')
                elif x < logo.shape[1] - 1:
                    file.write(',')
                if x == logo.shape[1] - 1 and y < logo.shape[0] - 1:
                    file.write('0x0A,')
            file.write('\n')
        file.write('};')

if __name__ == '__main__':
    main()