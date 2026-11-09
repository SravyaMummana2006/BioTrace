param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MavenArgs
)

# Resolve Java executable dynamically
$java = $null
if ($env:JAVA_HOME -and (Test-Path (Join-Path $env:JAVA_HOME "bin\java.exe"))) {
    $java = Join-Path $env:JAVA_HOME "bin\java.exe"
} elseif (Test-Path "C:\Program Files\Java\jdk-26.0.2\bin\java.exe") {
    $java = "C:\Program Files\Java\jdk-26.0.2\bin\java.exe"
} else {
    $cmd = Get-Command java -ErrorAction SilentlyContinue
    if ($cmd) {
        $java = $cmd.Source
    } else {
        $jdkFallback = Get-ChildItem "C:\Program Files\Java" -Filter "jdk-*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($jdkFallback -and (Test-Path (Join-Path $jdkFallback.FullName "bin\java.exe"))) {
            $java = Join-Path $jdkFallback.FullName "bin\java.exe"
        }
    }
}

if (-not $java) {
    Write-Error "Java executable not found. Please ensure Java 21+ is installed or set JAVA_HOME."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$m2Home = Join-Path $rootDir ".tools\apache-maven-3.9.6"
$launcherJar = Join-Path $m2Home "boot\plexus-classworlds-2.7.0.jar"
$conf = Join-Path $m2Home "bin\m2.conf"

& $java "-Dclassworlds.conf=$conf" "-Dmaven.home=$m2Home" "-Dmaven.multiModuleProjectDirectory=$rootDir\java-client" -classpath $launcherJar org.codehaus.plexus.classworlds.launcher.Launcher @MavenArgs

