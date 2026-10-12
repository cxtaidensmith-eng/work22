param(
    [string]$Python = "python",
    [int]$Cuda = 0
)

Set-Location -LiteralPath $PSScriptRoot

& $Python main.py --Config_name ./Config/main_tadpole_3class.ini --cuda $Cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python main.py --Config_name ./Config/main_tadpole_conversion.ini --cuda $Cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python main.py --Config_name ./Config/main_abide.ini --cuda $Cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python main.py --Config_name ./Config/main_abide5.ini --cuda $Cuda
exit $LASTEXITCODE
