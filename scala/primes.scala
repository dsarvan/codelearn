import scala.math._

object primes extends App {

    def prime(number: Int): String = {
        val sqrt_number: Int = sqrt(number).toInt

        for (i <- 2 to (sqrt_number + 1))
            if (number % i == 0) 
                return f"$number is not a prime number"
    
        return f"$number is a prime number"
    }


    for (n <- 4 to 30)
        println(prime(n))
}
