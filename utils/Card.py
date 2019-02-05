
class Card(object):
    def __init__(self, number):
        """
            Three attributes:
            Value is 1->A, 2->2, ... 10->10, 11->Jack, 12->Queen, 13-> King
            Type is 1->Spades, 2->Hearts, 3->Diamonds, 4->Clubs
            point is A->4, King->3, Queen->2, Jack->1
        """
        self.value = number % 13 + 1
        self.type = number // 13 + 1

        if self.value == 1:
            self.point = 4
        elif self.value == 13:
            self.point = 3
        elif self.value == 12:
            self.point = 2
        elif self.value == 11:
            self.point = 1
        else:
            self.point = 0

    def __repr__(self):
        four_type = ["Spades", 'Hearts', 'Diamonds', 'Clubs']
        return "{} of {}".format(self.value, four_type[self.type - 1])
