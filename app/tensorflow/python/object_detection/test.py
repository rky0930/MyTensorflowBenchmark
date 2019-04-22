class test:
    def __init__(self):
        pass 

    def tt(self):
        def inner():
            global v
            v = 100

        v = None
        print("1: ",v)
        inner()
        print("2:", v)

if __name__ == "__main__":
    t = test()

    t.tt()

