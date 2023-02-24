import cv2 as cv
import numpy as np

count = {}
shoe = {}


def initShoe(numDecks):
    shoe1 = {
        'ace': numDecks*4,
        'two': numDecks*4,
        'three': numDecks*4,
        'four': numDecks * 4,
        'five': numDecks * 4,
        'six': numDecks * 4,
        'seven': numDecks * 4,
        'eight': numDecks * 4,
        'nine': numDecks * 4,
        'ten': numDecks * 4,
        'jack': numDecks * 4,
        'queen': numDecks * 4,
        'king': numDecks * 4
    }
    return shoe1


def getCardsInShoe():
    numCards = 0
    for cardType in shoe:
        numCards += shoe[cardType]
    return numCards


def decrement(card):
    shoe[card] -= 1


# Calculates odds of busting on next card only
def calcBustOdds(hand, handVal):
    if handVal < 12 or isSoft(hand, handVal):
        return 0
    else:
        # Check how many of the cards in the deck will result in a hand not busting(easier to iterate from the bottom)
        # This will be the number of cards less than 21-hand. Aces should be included. Divide that by total cards left
        bustValue = 22 - handVal
        i = 0
        goodHitCount = 0
        while i < bustValue:
            goodHitCount += shoe[cardIntToString(i)]
            i += 1
        totalCards = getCardsInShoe()
        pNotBust = goodHitCount / totalCards
        return 1 - pNotBust


def isSoft(hand, handValue):
    if 'ace' in hand:
        # If the hand contains an ace, it is soft if counting the ace as 11 would not bust the hand.
        return handValue <= 11 + 10 * (hand.count('ace') - 1)
    else:
        # If the hand does not contain an ace, it is hard.
        return False


def calcHandVal(hand):
    # Initialize the hand value to 0.
    hand_value = 0

    # Count the number of aces in the hand.
    num_aces = hand.count('ace')

    # Iterate through the hand and add up the card values.
    for card in hand:
        if card == 'ace':
            # Aces are worth 11 by default, but can be reduced to 1 if needed.
            card_value = 11
        elif card in ['king', 'queen', 'jack']:
            # Face cards are worth 10.
            card_value = 10
        else:
            # All other cards are worth their face value.
            card_value = cardStringToInt(card)

        hand_value += card_value

    # If the hand contains aces and the hand value is over 21, reduce the value of the aces to 1.
    while num_aces > 0 and hand_value > 21:
        hand_value -= 10
        num_aces -= 1

    return hand_value


def cardStringToInt(cardString):
    valMap = {
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10
    }
    return valMap[cardString]


def cardIntToString(cardInt):
    valMap = {
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten'
    }
    return valMap[cardInt]
