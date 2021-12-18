class Rectangle (val x1 : Int, val y1 : Int, val x2 : Int, val y2 : Int) {

  def width = this.x2 - this.x1

  def height = this.y2 - this.y2

  def area = this.width * this.height

}

object RectangleApp {

  val rect1 = new Rectangle(0, 0, 10, 10)

  println(rect1.area)

  val rect2 = new Rectangle(5, 5, 10, 10)

  println(rect2.area)

}
