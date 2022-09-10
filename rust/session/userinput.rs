use std::io;

fn main() {

    let mut num = String::new();

    println!("Enter input: ");
    io::stdin().read_line(&mut num).expect("Invalid input!");

    println!("{}", num);

    let n: u32 = 11;

    for n in 1..n {
        println!("{}", n);
    }

}
