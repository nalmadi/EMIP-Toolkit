public class Rectangle {

	private int x1 , y1 , x2 , y2 ;

	public Rectangle ( int x1 , int y1 , int x2 , int y2 ) {

		this.x1 = x1 ;

		this.y1 = y1 ;

		this.x2 = x2 ;

		this.y2 = y2 ;

	}

	public int width ( ) { return this.x2 - this.x1 ; }

	public int height ( ) { return this.y2 - this.y1 ; }

	public double area ( ) { return this.width ( ) * this.height ( ) ; }

	public static void main ( String [ ] args ) {

		Rectangle rect1 = new Rectangle ( 0 , 0 , 10 , 10 ) ;

		System.out.println ( rect1.area ( ) ) ;

		Rectangle rect2 = new Rectangle ( 5 , 5 , 10 , 10 ) ;

		System.out.println ( rect2.area ( ) ) ;

	}

}

