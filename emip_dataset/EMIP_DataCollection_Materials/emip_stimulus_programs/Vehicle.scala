class Vehicle (p: String, t: String, tp: Int){
  
  val producer = p
  val vtype = t
  val topSpeed = tp
  var currentSpeed = 0
  
  def accelerate(kmh: Int) = {
    if ((this.currentSpeed + kmh) > this.topSpeed) {
      this.currentSpeed = this.topSpeed
    } else {
      this.currentSpeed = this.currentSpeed + kmh
    }    
    this.currentSpeed
  }
  
}

object VehicleApp extends App {
  val v = new Vehicle("Audi", "A6", 200)
  v.accelerate(10)
}