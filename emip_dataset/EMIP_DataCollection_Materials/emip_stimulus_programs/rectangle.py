class Rectangle :

    def __init__ (self , x1 , y1 , x2 , y2 ) :

        self.x1 = x1

        self.y1 = y1

        self.x2 = x2

        self.y2 = y2

    def width ( self ) : return self.x2 - self.x1

    def height ( self ) : return self.y2 - self.y1

    def area ( self ) : return self.width ( ) * self.height ( )

rect1 = Rectangle ( 0 , 0 , 10 , 10 )

print rect1.area ( )

rect2 = Rectangle ( 5 , 5 , 10 , 10 )

print rect2.area ( )
