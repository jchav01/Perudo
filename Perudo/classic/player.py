import random
from collections import Counter

class Player:
    def __init__(self, name, strategy='prudent'):
        self.name = name
        self.dice = [random.randint(1, 6) for _ in range(5)]
        self.eliminated = False
        self.strategy = strategy
        self.proposed_bid = None

    def roll_dice(self):
        if not self.eliminated:
            self.dice = [random.randint(1, 6) for _ in range(len(self.dice))]

    def lose_die(self):
        if self.dice:
            self.dice.pop()
            if not self.dice:
                self.eliminate()

    def has_dice(self):
        return len(self.dice) > 0 and not self.eliminated

    def eliminate(self):
        self.eliminated = True

    def get_dice(self):
        return self.dice
    
    def organize_dice(self):
        dice_count = Counter(self.dice)
        organized_dice = [(count, value) for value, count in dice_count.items()]
        organized_dice.sort(key=lambda x: x[1])
        return organized_dice

    def best_bet(self):
        max = 0
        best = (0,0)
        organized_dice = self.organize_dice()
        for k in organized_dice:
            if k[0] >= max:
                max = k[0]
                best = k
        
        return best
                

    def bet(self, current_bid, total_dice):
        if self.strategy == 'shy':
            return self.shy_bet(current_bid)
        elif self.strategy == 'prudent':
            return self.prudent_bet(current_bid, total_dice)
        elif self.strategy == 'brusque':
            return self.brusque_bet(current_bid, total_dice)
        elif self.strategy == 'bluffeur':
            return self.bluffeur_bet(current_bid, total_dice)
        else:
            raise ValueError("Unknown strategy")

    def shy_bet(self, current_bid):
        # This guy is sure that he won't be called a liar
        self.proposed_bid = (0,0)
        
        for k in self.organize_dice():
            if k[1] > current_bid[1] and k[0] >= current_bid[0]:
                    self.proposed_bid = (current_bid[0], k[1])
                    break
        else:
            for k in self.organize_dice():
                if k[0] > current_bid[0]:
                    self.proposed_bid = (current_bid[0] + 1, k[1])
                    break
                
        print(f"{self.name} (Shy) fait un pari: {self.proposed_bid}")
        return self.proposed_bid
    
    def prudent_bet(self, current_bid, total_dice):
        # This guy might lie, but only if it is fair to think he can (based on his dice, and the probability of the current_bid - based on total dice)
        # His strategy of incrementation is the same as the shy guy
        
        self.proposed_bid = (0,0)
        for k in self.organize_dice():
            if k[1] > current_bid[1] and k[0] + int(1/6 * total_dice) >= current_bid[0]:
                    self.proposed_bid = (current_bid[0], k[1])
                    break
        else:
            for k in self.organize_dice():
                if k[0] + int(1/6 * total_dice) > current_bid[0]:
                    self.proposed_bid = (current_bid[0] + 1, k[1])
                    break
       
        print(f"{self.name} (Prudent) fait un pari: {self.proposed_bid}")
        return self.proposed_bid

    def brusque_bet(self, current_bid, total_dice):
        
        self.proposed_bid = (0,0)
        
        best = self.best_bet()
        if current_bid[0] < best[0]:
            self.proposed_bid = (best[0], best[1])
        elif current_bid[0] < best[0] + 1/6*total_dice and current_bid[1] <= best[1]:
            self.proposed_bid = (current_bid[0] + 1, best[1])
        elif current_bid[0] < best[0] + 1/6*total_dice and current_bid[1] > best[1]:
            self.proposed_bid = (current_bid[0] + 1, current_bid[1])
        
        print(f"{self.name} (Brusque) fait un pari: {self.proposed_bid}")
        return self.proposed_bid

    def bluffeur_bet(self, current_bid, total_dice):
        if current_bid[0] > total_dice:
            self.proposed_bid = (0,0)
            return self.proposed_bid
        
        self.proposed_bid = (current_bid[0], 6) if current_bid[1] != 6 else (current_bid[0] + 1, current_bid[1])
        print(f"{self.name} (Bluffeur) fait un pari: {self.proposed_bid}")
        return self.proposed_bid
