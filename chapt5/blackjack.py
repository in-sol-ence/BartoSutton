from dataclasses import dataclass
import random

class state:
    def __init__(self):
        self.playerCards = [self.getCard(), self.getCard()]
        self.dealerCards = [self.getCard(), self.getCard()]
        self.viewableDealerCard = self.dealerCards[0]
        self._terminal = False

    def getCard(self) -> int:
        """Helper function to get a card"""
        card = random.randint(1, 13)
        if card > 10:
            return 10
        elif card == 1:
            return 11 
        else:
            return card
    
    @property
    def isPlayerUsuableAce(self) -> bool:
        for card in self.playerCards:
            if card == 11:
                return True
        return False

    @property
    def isDealerUsuableAce(self) -> bool:
        for card in self.dealerCards:
            if card == 11:
                return True
        return False

    @property
    def playerSum(self) -> int:
        if self.isPlayerUsuableAce and sum(self.playerCards) > 21:
            total = sum(self.playerCards) - 10
        else:
            total = sum(self.playerCards)
        return total

    @property
    def dealerSum(self) -> int:
        if self.isDealerUsuableAce and sum(self.dealerCards) > 21:
            total = sum(self.dealerCards) - 10
        else:
            total = sum(self.dealerCards)
        return total

    @property
    def terminal(self) -> bool:
        if self.playerSum > 21:
            return True
        if self.dealerSum >= 17:
            return True
        return False

    @terminal.setter
    def terminal(self, value: bool):
        self._terminal = value

class blackJack:
    def __init__(self) -> state:
        self._state = state()
        return None
    
    def hit(self) -> state:
        if not self._state.terminal:
            self._state.playerCards.append(self.state.getCard())
            return self._state
        else:
            raise Exception("Game is already over.")

    def stick(self) -> state:
        if not self._state.terminal:
            while self._state.dealerSum < 17:
                self._state.dealerCards.append(self.state.getCard())
            self._state.terminal = True
            return self._state
        else:
            raise Exception("Game is already over.")
    
    @property
    def state(self) -> state:
        return self._state
    
    @property
    def reward(self) -> int:
        """The reward is set with gamma = 1 and the only reward is at the terminal state."""
        if self._state.playerSum > 21:
            return -1
        if self._state.dealerSum==21:
            if self._state.playerSum==21:
                return 0
            return -1
        if self._state.playerSum==21:
            return 1
        return 0
    
if __name__ == "__main__":
    game = blackJack()
    print("Initial State:")
    print("Player Cards:", game.state.playerCards, "Sum:", game.state.playerSum)
    print("Dealer viewable Card:", game.state.viewableDealerCard)
    while not game.state.terminal:
        action = input("Enter 'h' to hit or 's' to stick: ")
        if action == 'h':
            game.hit()
            print("Player Cards:", game.state.playerCards, "Sum:", game.state.playerSum)
        elif action == 's':
            game.stick()
        else:
            print("Invalid action. Please enter 'h' or 's'.")
    print("Final State:")
    print("Player Cards:", game.state.playerCards, "Sum:", game.state.playerSum)
    print("Dealer Cards:", game.state.dealerCards, "Sum:", game.state.dealerSum)
    print("Reward:", game.reward)