use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use sysinfo::{RefreshKind, System, SystemExt, ProcessRefreshKind, Pid, ProcessExt};
use std::fs::File;
use std::io::Write;

fn monitor() {
    std::thread::spawn(|| {
        let filename = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis().to_string();
        let mut outputfile = File::create(filename).unwrap();

        let pid = Pid::from(std::process::id() as usize);
        let mut sys = System::new_with_specifics(
            RefreshKind::new().with_processes(ProcessRefreshKind::everything())
        );
        
        let interval = Duration::from_millis(1000);
        assert!(System::MINIMUM_CPU_UPDATE_INTERVAL < interval);

        sys.refresh_process(pid);
        loop {
            std::thread::sleep(interval);
            sys.refresh_process(pid);
            if let Some(process) = sys.process(pid) {
                writeln!(outputfile, "{}% {} bytes",process.cpu_usage(), process.memory()).unwrap();
            }
        }
    });

    loop {
        let mut sum: usize = 0;
        for i in 0..10000 {
            sum += i;
        }
    }
}