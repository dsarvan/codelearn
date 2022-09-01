fn main() {
    // closure
    let fizzbuzz = |x| {
        if x % 15 == 0 {
            println!("FizzBuzz");
        } else if x % 3 == 0 {
            println!("Fizz");
        } else if x % 5 == 0 {
            println!("Buzz");
        } else {
            println!("{}", x);
        }
    };

    // function
    fn fizzbuzz(x: i32) {
        if x % 15 == 0 {
            println!("FizzBuzz");
        } else if x % 3 == 0 {
            println!("Fizz");
        } else if x % 5 == 0 {
            println!("Buzz");
        } else {
            println!("{}", x);
        }
    }

    for n in 1..16 {
        fizzbuzz(n)
    }

    println!("---");
    (1..16).into_iter().for_each(fizzbuzz); // pass function fizzbuzz as arg
}
