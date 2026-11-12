$ErrorActionPreference = "Stop"

$deps = Get-ChildItem "$env:USERPROFILE\.m2\repository" -Recurse -Filter *.jar |
    Where-Object {
        $_.FullName -match '\\(com\\formdev\\flatlaf|org\\jfree\\jfreechart|org\\openpnp\\opencv)\\'
    } |
    ForEach-Object { $_.FullName }

$classpath = "target\classes;" + ($deps -join ";")

java -cp $classpath com.biotrace.app.BioTraceApp
