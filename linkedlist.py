class Node():
    def __init__(self,data=None,Lchild=None,Rchild=None):
        self.data= data
        self.Lchild= Lchild
        self.Rchild = Rchild
class Tree():
    def __init__(self,root=None):
        self.root = root
    def insert(self,val):

        if(self.root is None):
            self.root = Node(val)
            print('Root Added')
        else:
            start = self.root
            queue=[]
            queue.append(start)
            while(len(queue) > 0):
                newNode =Node(val)
                pointer = queue.pop(0)
                if pointer.Lchild is None:
                    pointer.Lchild = newNode
                    print("new node added as left child")
                    break
                elif pointer.Rchild is None:
                    pointer.Rchild = newNode
                    print("new node added as right child")
                    break
                if pointer.Lchild != None:
                    queue.append(pointer.Lchild)
                if pointer.Rchild != None:
                    queue.append(pointer.Rchild)
    def display(self):
        start=self.root
        queue=[]
        queue.append(start)
        while(len(queue)>0):
            pointer=queue.pop(0)
            print(pointer.data)
            if(pointer.Lchild != None):
                queue.append(pointer.Lchild)
            if(pointer.Rchild != None):
                queue.append(pointer.Rchild)
        print("end of display")
t= Tree()
t.insert(1)
t.insert(2)
t.insert(3)
t.insert(4)
t.insert(5)
t.insert(6)
t.insert
t.display()