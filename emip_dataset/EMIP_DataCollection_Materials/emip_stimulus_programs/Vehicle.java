public class Vehicle {

	String producer , type ;

	int topSpeed , currentSpeed ;


	public Vehicle ( String p , String t , int tp ) {

		this.producer = p ;

		this.type = t ;

		this.topSpeed = tp ;

		this.currentSpeed = 0 ;

	}


	public int accelerate ( int kmh ) {

		if ( ( this.currentSpeed + kmh ) > this.topSpeed ) {

			this.currentSpeed = this.topSpeed ;

		} else {
	
			this.currentSpeed = this.currentSpeed + kmh ;

		}

		return this.currentSpeed ;

	}


	public static void main ( String args [ ] ) {

		Vehicle v = new Vehicle ( "Audi" , "A6" , 200 ) ;

		v.accelerate ( 10 ) ;

	}

}
