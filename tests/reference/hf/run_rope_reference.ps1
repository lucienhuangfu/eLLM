param(
    [string]$Case = "$PSScriptRoot\cases\rope_case_min.json",
    [string]$Output = "$PSScriptRoot\golden\rope_case_min.json"
)

$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $launcher) {
    throw "Python launcher 'py' was not found. Install Python or run rope.py directly."
}

& py -3 "$PSScriptRoot\rope.py" --case $Case --output $Output
