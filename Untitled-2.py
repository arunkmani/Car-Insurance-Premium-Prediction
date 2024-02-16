class Node:
    def __init__(self,data=None,next=None):
        self.data=data
        self.next =None
class linkedlist:
    def __init__(self):
        self.head= None
    def insert(self,data):
        newnode = Node(data)
        if self.head == None:
            self.head = newnode
        else:
            pointer = self.head
            while(pointer.next):
                pointer = pointer.next
            pointer.next=newnode
    def display(self):
        pointer = self.head
        while(pointer):
            print(pointer.data)
            pointer = pointer.next
ll=linkedlist()
ll.insert(2)
ll.insert(4)
ll.insert(6)
ll.insert(8)
ll.insert(10)
ll.insert(12)
ll.insert(14)
ll.insert(16)
ll.insert(18)
ll2=linkedlist()
ll2.insert(1)
ll2.insert(3)
ll2.insert(5)
ll2.insert(7)
ll2.insert(9)
ll2.insert(11)
ll2.insert(13)
ll2.insert(15)
ll2.insert(17)
final=linkedlist()
print(ll.head.data)
print(ll2.head.data)
pointer1=ll.head
pointer2=ll2.head
while(pointer1 and pointer2):
    if(pointer1.data<pointer2.data):
        final.insert(pointer1.data)
        pointer1=pointer1.next
    elif(pointer1.data>pointer2.data):
        final.insert(pointer2.data)
        pointer2=pointer2.next
    elif(pointer1.data == pointer2.data):
        final.insert(pointer1.data)
        pointer1=pointer1.next
        pointer2=pointer2.next
print('Printing final')
final.display()


            
