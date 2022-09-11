object factorial extends App {
    def factorial(n: BigInt): BigInt = 
        if (n == 0) 1 else n * factorial(n - 1)

    val N: Int = 10
    println(factorial(N))
}
