param(
    [switch]$Headless
)

$java = "C:\Program Files\Java\jdk-26.0.2\bin\java.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$targetClasses = Join-Path $rootDir "java-client\target\classes"
$m2Repo = "C:\Users\Sravya mummana\.m2\repository"

$jars = @(
    "$targetClasses",
    (Join-Path $m2Repo "com\formdev\flatlaf\3.4.1\flatlaf-3.4.1.jar"),
    (Join-Path $m2Repo "org\jfree\jfreechart\1.5.4\jfreechart-1.5.4.jar"),
    (Join-Path $m2Repo "org\slf4j\slf4j-api\1.7.36\slf4j-api-1.7.36.jar"),
    (Join-Path $m2Repo "commons-io\commons-io\2.15.1\commons-io-2.15.1.jar")
)

$cp = $jars -join ";"

$argsList = @("-cp", $cp)
if ($Headless) {
    $argsList += "-Djava.awt.headless=true"
}
$argsList += "org.biotrace.client.BioTraceApplication"

Write-Host "Launching BIOTrace Forensic Desktop Application..." -ForegroundColor Cyan
Write-Host "Classpath: $cp" -ForegroundColor DarkGray

& $java @argsList
