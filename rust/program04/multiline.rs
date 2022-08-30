fn main() {
    let more = "many
lines
    whitespace"
        .to_string(); // create string
    println!("{}", more);

    let escape = " blah \"\"\" escaped '
newline
\n
test
"
    .to_string();
    println!("{}", escape);

    let _single_char = 'a';
}
