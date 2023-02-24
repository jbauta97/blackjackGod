import pyautogui
import cv2 as cv
import numpy as np
from time import time
import pyscreenshot as ImageGrab
import pygetwindow as gw
import mss
import mss.tools


# Return number of card found
def hasCard(ref, imgLightBlack, imgLightRed, imgDarkBlack, imgDarkRed, name):
    playerResultBlack = cv.matchTemplate(ref, imgDarkBlack, cv.TM_CCOEFF_NORMED)
    playerResultRed = cv.matchTemplate(ref, imgDarkRed, cv.TM_CCOEFF_NORMED)
    dealerResultBlack = cv.matchTemplate(ref, imgLightBlack, cv.TM_CCOEFF_NORMED)
    dealerResultRed = cv.matchTemplate(ref, imgLightRed, cv.TM_CCOEFF_NORMED)
    count = 0
    min_vPB, max_vPB, min_lPB, max_lPB = cv.minMaxLoc(playerResultBlack)
    min_vPR, max_vPR, min_lPR, max_lPR = cv.minMaxLoc(playerResultRed)
    min_vDB, max_vDB, min_lDB, max_lDB = cv.minMaxLoc(dealerResultBlack)
    min_vDR, max_vDR, min_lDR, max_lDR = cv.minMaxLoc(dealerResultRed)
    confidence = .97
    if foundImage(playerResultBlack, confidence):
        wid = imgDarkBlack.shape[1]
        hei = imgDarkBlack.shape[0]
        bot_r = (max_lPB[0] + wid, max_lPB[1] + hei)
        cv.rectangle(ref, max_lPB, bot_r, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        count += 1
        print('Player has: ' + name + '      conf: ' + str(max_vPB))
    if foundImage(playerResultRed, confidence):
        wid = imgDarkRed.shape[1]
        hei = imgDarkRed.shape[0]
        bot_r = (max_lPR[0] + wid, max_lPR[1] + hei)
        cv.rectangle(ref, max_lPR, bot_r, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        count += 1
        print('Player has: ' + name + '      conf: ' + str(max_vPR))
    if foundImage(dealerResultBlack, confidence):
        wid = imgLightBlack.shape[1]
        hei = imgLightBlack.shape[0]
        bot_r = (max_lDB[0] + wid, max_lDB[1] + hei)
        cv.rectangle(ref, max_lDB, bot_r, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        count += 1
        print('Dealer has: ' + name + '      conf: ' + str(max_vDB))
    if foundImage(dealerResultRed, confidence):
        wid = imgLightRed.shape[1]
        hei = imgLightRed.shape[0]
        bot_r = (max_lDR[0] + wid, max_lDR[1] + hei)
        cv.rectangle(ref, max_lDR, bot_r, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
        count += 1
        print('Dealer has: ' + name + '      conf: ' + str(max_vDR))
    return count


def foundImage(matchedResult, threshold):
    min_v, max_v, min_l, max_l = cv.minMaxLoc(matchedResult)
    return max_v > threshold


window = gw.getWindowsWithTitle('Draftkings')[0]
window.activate()
winX, winY, winWid, winHei = window.left, window.top, window.width, window.height
monitor = {"top": winY+400, "left": winX, "width": winWid-650, "height": winHei-800}

imAceBlack = cv.imread('trainingData\\acePBlack.png', cv.IMREAD_GRAYSCALE)
imAceRed = cv.imread('trainingData\\acePRed.png', cv.IMREAD_GRAYSCALE)
imAceDealerBlack = cv.imread('trainingData\\aceDBlack.png', cv.IMREAD_GRAYSCALE)
imAceDealerRed = cv.imread('trainingData\\aceDRed.png', cv.IMREAD_GRAYSCALE)
imTwoBlack = cv.imread('trainingData\\2PBlack.png', cv.IMREAD_GRAYSCALE)
imTwoRed = cv.imread('trainingData\\2PRed.png', cv.IMREAD_GRAYSCALE)
imTwoDealerBlack = cv.imread('trainingData\\2DBlack.png', cv.IMREAD_GRAYSCALE)
imTwoDealerRed = cv.imread('trainingData\\2DRed.png', cv.IMREAD_GRAYSCALE)
imThreeBlack = cv.imread('trainingData\\3PBlack.png', cv.IMREAD_GRAYSCALE)
imThreeRed = cv.imread('trainingData\\3PRed.png', cv.IMREAD_GRAYSCALE)
imThreeDealerBlack = cv.imread('trainingData\\3DBlack.png', cv.IMREAD_GRAYSCALE)
imThreeDealerRed = cv.imread('trainingData\\3DRed.png', cv.IMREAD_GRAYSCALE)
imFourBlack = cv.imread('trainingData\\4PBlack.png', cv.IMREAD_GRAYSCALE)
imFourRed = cv.imread('trainingData\\4PRed.png', cv.IMREAD_GRAYSCALE)
imFourDealerBlack = cv.imread('trainingData\\4DBlack.png', cv.IMREAD_GRAYSCALE)
imFourDealerRed = cv.imread('trainingData\\4DRed.png', cv.IMREAD_GRAYSCALE)
imFiveBlack = cv.imread('trainingData\\5PBlack.png', cv.IMREAD_GRAYSCALE)
imFiveRed = cv.imread('trainingData\\5PRed.png', cv.IMREAD_GRAYSCALE)
imFiveDealerBlack = cv.imread('trainingData\\5DBlack.png', cv.IMREAD_GRAYSCALE)
imFiveDealerRed = cv.imread('trainingData\\5DRed.png', cv.IMREAD_GRAYSCALE)
imSixBlack = cv.imread('trainingData\\6PBlack.png', cv.IMREAD_GRAYSCALE)
imSixRed = cv.imread('trainingData\\6PRed.png', cv.IMREAD_GRAYSCALE)
imSixDealerBlack = cv.imread('trainingData\\6DBlack.png', cv.IMREAD_GRAYSCALE)
imSixDealerRed = cv.imread('trainingData\\6DRed.png', cv.IMREAD_GRAYSCALE)
imSevenBlack = cv.imread('trainingData\\7PBlack.png', cv.IMREAD_GRAYSCALE)
imSevenRed = cv.imread('trainingData\\7PRed.png', cv.IMREAD_GRAYSCALE)
imSevenDealerBlack = cv.imread('trainingData\\7DBlack.png', cv.IMREAD_GRAYSCALE)
imSevenDealerRed = cv.imread('trainingData\\7DRed.png', cv.IMREAD_GRAYSCALE)
imEightBlack = cv.imread('trainingData\\8PBlack.png', cv.IMREAD_GRAYSCALE)
imEightRed = cv.imread('trainingData\\8PRed.png', cv.IMREAD_GRAYSCALE)
imEightDealerBlack = cv.imread('trainingData\\8DBlack.png', cv.IMREAD_GRAYSCALE)
imEightDealerRed = cv.imread('trainingData\\8DRed.png', cv.IMREAD_GRAYSCALE)
imNineBlack = cv.imread('trainingData\\9PBlack.png', cv.IMREAD_GRAYSCALE)
imNineRed = cv.imread('trainingData\\9PRed.png', cv.IMREAD_GRAYSCALE)
imNineDealerBlack = cv.imread('trainingData\\9DBlack.png', cv.IMREAD_GRAYSCALE)
imNineDealerRed = cv.imread('trainingData\\9DRed.png', cv.IMREAD_GRAYSCALE)
imTenBlack = cv.imread('trainingData\\10PBlack.png', cv.IMREAD_GRAYSCALE)
imTenRed = cv.imread('trainingData\\10PRed.png', cv.IMREAD_GRAYSCALE)
imTenDealerBlack = cv.imread('trainingData\\10DBlack.png', cv.IMREAD_GRAYSCALE)
imTenDealerRed = cv.imread('trainingData\\10DRed.png', cv.IMREAD_GRAYSCALE)
imJackBlack = cv.imread('trainingData\\jackPBlack.png', cv.IMREAD_GRAYSCALE)
imJackRed = cv.imread('trainingData\\jackPRed.png', cv.IMREAD_GRAYSCALE)
imJackDealerBlack = cv.imread('trainingData\\jackDBlack.png', cv.IMREAD_GRAYSCALE)
imJackDealerRed = cv.imread('trainingData\\jackDRed.png', cv.IMREAD_GRAYSCALE)
imQueenBlack = cv.imread('trainingData\\queenPBlack.png', cv.IMREAD_GRAYSCALE)
imQueenRed = cv.imread('trainingData\\queenPRed.png', cv.IMREAD_GRAYSCALE)
imQueenDealerBlack = cv.imread('trainingData\\queenDBlack.png', cv.IMREAD_GRAYSCALE)
imQueenDealerRed = cv.imread('trainingData\\queenDRed.png', cv.IMREAD_GRAYSCALE)
imKingBlack = cv.imread('trainingData\\kingPBlack.png', cv.IMREAD_GRAYSCALE)
imKingRed = cv.imread('trainingData\\kingPRed.png', cv.IMREAD_GRAYSCALE)
imKingDealerBlack = cv.imread('trainingData\\kingDBlack.png', cv.IMREAD_GRAYSCALE)
imKingDealerRed = cv.imread('trainingData\\kingDRed.png', cv.IMREAD_GRAYSCALE)


with mss.mss() as sct:

    loop_time = time()
    while True:
        #im1 = pyautogui.screenshot()
        #im1 = ImageGrab.grab(bbox)
        #im1.save(r'trainingData\img.png')
        im1 = sct.grab(monitor)

        im1_cv = np.array(im1)
        im1_cv = cv.cvtColor(im1_cv, cv.COLOR_BGR2GRAY)

        hasCard(im1_cv, imAceDealerBlack, imAceDealerRed, imAceBlack, imAceRed, 'ace')
        hasCard(im1_cv, imTwoDealerBlack, imTwoDealerRed, imTwoBlack, imTwoRed, 'two')
        hasCard(im1_cv, imThreeDealerBlack, imThreeDealerRed, imThreeBlack, imThreeRed, 'three')
        hasCard(im1_cv, imFourDealerBlack, imFourDealerRed, imFourBlack, imFourRed, 'four')
        hasCard(im1_cv, imFiveDealerBlack, imFiveDealerRed, imFiveBlack, imFiveRed, 'five')
        hasCard(im1_cv, imSixDealerBlack, imSixDealerRed, imSixBlack, imSixRed, 'six')
        hasCard(im1_cv, imSevenDealerBlack, imSevenDealerRed, imSevenBlack, imSevenRed, 'seven')
        hasCard(im1_cv, imEightDealerBlack, imEightDealerRed, imEightBlack, imEightRed, 'eight')
        hasCard(im1_cv, imNineDealerBlack, imNineDealerRed, imNineBlack, imNineRed, 'nine')
        hasCard(im1_cv, imTenDealerBlack, imTenDealerRed, imTenBlack, imTenRed, 'ten')
        hasCard(im1_cv, imJackDealerBlack, imJackDealerRed, imJackBlack, imJackRed, 'jack')
        hasCard(im1_cv, imQueenDealerBlack, imQueenDealerRed, imQueenBlack, imQueenRed, 'queen')
        hasCard(im1_cv, imKingDealerBlack, imKingDealerRed, imKingBlack, imKingRed, 'king')

        fps = 'FPS {}'.format(1 / (time() - loop_time))
        #print(fps)
        cv.imshow('testwindow', im1_cv)
        loop_time = time()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


