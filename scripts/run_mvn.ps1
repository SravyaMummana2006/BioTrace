param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MavenArgs
)

$java = "C:\Program Files\Java\jdk-26.0.2\bin\java.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$m2Home = Join-Path $rootDir ".tools\apache-maven-3.9.6"
$launcherJar = Join-Path $m2Home "boot\plexus-classworlds-2.7.0.jar"
$conf = Join-Path $m2Home "bin\m2.conf"

& $java "-Dclassworlds.conf=$conf" "-Dmaven.home=$m2Home" "-Dmaven.multiModuleProjectDirectory=$rootDir\java-client" -classpath $launcherJar org.codehaus.plexus.classworlds.launcher.Launcher @MavenArgs
