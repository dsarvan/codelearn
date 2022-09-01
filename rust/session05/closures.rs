fn main() {
    let example_closure = |x| x;

    let s = example_closure(String::from("session"));
    //let n = example_closure(5); // error since rust assumes example_closure
    //takes string input because earlier we passed a string as argument to the
    // example_closure() function
    let n = example_closure(5.to_string());
    println!("{}, {}", s, n);

    let x = vec![1, 2, 3];
    let equal_to_x = move |z| z == x; // x moved to the scope of this closure
    // no longer print x since it is not in scope main() because of equal_to_x
    //println!("This is x {:?}", x); 
    let y = vec![1, 2, 3];
    assert!(equal_to_x(y));

    let items: Vec<i32> = vec![1, 2, 3, 4, 5];
    let plus_one: Vec<_> = items.iter().map(|x| x + 1).collect();
    let sum_all: i32 = items.iter().map(|x| x + 1).sum();
    println!("{:?}, {}", plus_one, sum_all);

    let two_args = |x, y| x - y;
    println!("{}", two_args(5, 3));
}
