class Vehicle :    

    def __init__ ( self , p , t , tp ) :

        self.producer = p

        self.vtype = t

        self.topSpeed = tp 

        self.currentSpeed = 0
        

    def accelerate ( self , kmh ) :

        if ( self.currentSpeed + kmh ) > self.topSpeed :

            self.currentSpeed = self.topSpeed

        else :

            self.currentSpeed = self.currentSpeed + kmh

        return self.currentSpeed   


v = Vehicle ( "Audi" , "A6" , 200 )

v.accelerate ( 10 )

