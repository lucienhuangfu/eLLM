use std::io::Write;
fn prompt(name: &str) -> String {
    let mut line = String::new();
    print!("{}", name);
    std::io::stdout().flush().unwrap();
    std::io::stdin()
        .read_line(&mut line)
        .expect("Error: Could not read a line");

    return line.trim().to_string();
}
/* 
fn main() {
    loop {
        let input = prompt("> ");
        if input == "exit" {
            break;
        } else {
            // let unixtime = SystemTime::now()
            //    .duration_since(SystemTime::UNIX_EPOCH)
            //    .unwrap();
            // print!("Current Unix time is {:?}\n", unixtime);
        };
    }
}
*/