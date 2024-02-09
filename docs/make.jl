using Documenter
using SiriusTrack

makedocs(
    sitename="SiriusTrack",
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "Constants" => "about/constants.md",
            ]
    ],
    modules = [SiriusTrack]
)