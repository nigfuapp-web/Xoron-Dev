#!/usr/bin/env python3
"""
Synthetic dataset generator for system administration and language environment setup.

Generates training data for:
1. System package installation (apt, yum, brew, etc.)
2. Programming language environment setup (Swift, Java, Go, Rust, etc.)
3. Desktop environment setup (XFCE, GNOME, etc.)
4. SSH and remote access
5. Docker and containerization
6. Web servers and services
7. Database setup
8. Network configuration
9. File downloads and management
10. System monitoring and debugging

Usage:
    python -m synth.system_admin_generator
"""

import os
import sys
import json
import random
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.special_tokens import SPECIAL_TOKENS

# =============================================================================
# SYSTEM PACKAGE INSTALLATION
# =============================================================================

APT_PACKAGES = [
    {
        "task": "Install Chrome browser",
        "steps": [
            {"cmd": "wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -", "output": "OK"},
            {"cmd": "echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list", "output": "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main"},
            {"cmd": "sudo apt update", "output": "Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease\nGet:2 http://dl.google.com/linux/chrome/deb stable InRelease\nReading package lists... Done"},
            {"cmd": "sudo apt install -y google-chrome-stable", "output": "Reading package lists... Done\nSetting up google-chrome-stable (120.0.6099.109-1) ..."},
            {"cmd": "google-chrome --version", "output": "Google Chrome 120.0.6099.109"},
        ]
    },
    {
        "task": "Install Firefox browser",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y firefox", "output": "Reading package lists... Done\nSetting up firefox (121.0+build1-0ubuntu0.22.04.1) ..."},
            {"cmd": "firefox --version", "output": "Mozilla Firefox 121.0"},
        ]
    },
    {
        "task": "Install VS Code",
        "steps": [
            {"cmd": "wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg", "output": ""},
            {"cmd": "sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg", "output": ""},
            {"cmd": "sudo sh -c 'echo \"deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main\" > /etc/apt/sources.list.d/vscode.list'", "output": ""},
            {"cmd": "sudo apt update && sudo apt install -y code", "output": "Setting up code (1.85.1-1702462158) ..."},
            {"cmd": "code --version", "output": "1.85.1\n0ee08df0cf4527e40edc9aa28f4b5bd38bbff2b2\nx64"},
        ]
    },
    {
        "task": "Install build essentials and development tools",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y build-essential gcc g++ make cmake autoconf automake libtool pkg-config", "output": "Setting up build-essential (12.9ubuntu3) ..."},
            {"cmd": "gcc --version", "output": "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"},
            {"cmd": "make --version", "output": "GNU Make 4.3"},
            {"cmd": "cmake --version", "output": "cmake version 3.22.1"},
        ]
    },
    {
        "task": "Install curl, wget, and common utilities",
        "steps": [
            {"cmd": "sudo apt install -y curl wget git vim nano htop tree jq unzip zip tar gzip", "output": "Setting up curl (7.81.0-1ubuntu1.15) ...\nSetting up wget (1.21.2-2ubuntu1) ..."},
            {"cmd": "curl --version | head -1", "output": "curl 7.81.0 (x86_64-pc-linux-gnu)"},
            {"cmd": "wget --version | head -1", "output": "GNU Wget 1.21.2 built on linux-gnu."},
        ]
    },
    {
        "task": "Install network tools",
        "steps": [
            {"cmd": "sudo apt install -y net-tools iputils-ping traceroute nmap netcat dnsutils whois tcpdump wireshark-common", "output": "Setting up net-tools (1.60+git20181103.0eebece-1ubuntu5) ..."},
            {"cmd": "ifconfig --version 2>&1 | head -1", "output": "net-tools 2.10-alpha"},
            {"cmd": "nmap --version | head -1", "output": "Nmap version 7.80 ( https://nmap.org )"},
        ]
    },
    {
        "task": "Install multimedia codecs and tools",
        "steps": [
            {"cmd": "sudo apt install -y ubuntu-restricted-extras ffmpeg vlc", "output": "Setting up ffmpeg (7:4.4.2-0ubuntu0.22.04.1) ..."},
            {"cmd": "ffmpeg -version | head -1", "output": "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers"},
        ]
    },
]

YUM_PACKAGES = [
    {
        "task": "Install development tools on CentOS/RHEL",
        "steps": [
            {"cmd": "sudo yum groupinstall -y 'Development Tools'", "output": "Complete!"},
            {"cmd": "sudo yum install -y gcc gcc-c++ make cmake kernel-devel", "output": "Complete!"},
            {"cmd": "gcc --version | head -1", "output": "gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-4)"},
        ]
    },
    {
        "task": "Install EPEL repository and common tools",
        "steps": [
            {"cmd": "sudo yum install -y epel-release", "output": "Complete!"},
            {"cmd": "sudo yum install -y htop vim git curl wget jq", "output": "Complete!"},
        ]
    },
]

BREW_PACKAGES = [
    {
        "task": "Install Homebrew and common tools on macOS",
        "steps": [
            {"cmd": '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', "output": "==> Installation successful!"},
            {"cmd": "brew install git curl wget jq tree htop", "output": "==> Downloading https://ghcr.io/v2/homebrew/core/git/manifests/2.43.0\n==> Installing git\n🍺  /opt/homebrew/Cellar/git/2.43.0: 1,635 files, 51.3MB"},
            {"cmd": "brew --version", "output": "Homebrew 4.2.0"},
        ]
    },
]

# =============================================================================
# PROGRAMMING LANGUAGE ENVIRONMENTS
# =============================================================================

LANGUAGE_SETUPS = {
    "swift": {
        "name": "Swift",
        "description": "Set up Swift development environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y clang libicu-dev libcurl4-openssl-dev libssl-dev libxml2-dev", "output": "Setting up clang (1:14.0-55~exp2) ..."},
            {"cmd": "wget https://download.swift.org/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz", "output": "swift-5.9.2-RELEASE-ubuntu22.04.tar.gz  100%[===================>] 500M  25.0MB/s    in 20s"},
            {"cmd": "tar xzf swift-5.9.2-RELEASE-ubuntu22.04.tar.gz", "output": ""},
            {"cmd": "sudo mv swift-5.9.2-RELEASE-ubuntu22.04 /opt/swift", "output": ""},
            {"cmd": "echo 'export PATH=/opt/swift/usr/bin:$PATH' >> ~/.bashrc && source ~/.bashrc", "output": ""},
            {"cmd": "swift --version", "output": "Swift version 5.9.2 (swift-5.9.2-RELEASE)\nTarget: x86_64-unknown-linux-gnu"},
        ],
        "test_code": 'print("Hello, Swift!")',
        "test_cmd": "swift -e 'print(\"Hello, Swift!\")'",
        "test_output": "Hello, Swift!",
        "compile_cmd": "swiftc hello.swift -o hello && ./hello",
    },
    "java": {
        "name": "Java (OpenJDK)",
        "description": "Set up Java development environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y openjdk-17-jdk openjdk-17-jre", "output": "Setting up openjdk-17-jdk:amd64 (17.0.9+9-1~22.04) ..."},
            {"cmd": "java -version", "output": "openjdk version \"17.0.9\" 2023-10-17\nOpenJDK Runtime Environment (build 17.0.9+9-Ubuntu-122.04)\nOpenJDK 64-Bit Server VM (build 17.0.9+9-Ubuntu-122.04, mixed mode, sharing)"},
            {"cmd": "javac -version", "output": "javac 17.0.9"},
            {"cmd": "echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc", "output": ""},
            {"cmd": "echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc && source ~/.bashrc", "output": ""},
        ],
        "test_code": 'public class Hello { public static void main(String[] args) { System.out.println("Hello, Java!"); } }',
        "test_cmd": "javac Hello.java && java Hello",
        "test_output": "Hello, Java!",
    },
    "go": {
        "name": "Go (Golang)",
        "description": "Set up Go development environment",
        "steps": [
            {"cmd": "wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz", "output": "go1.21.5.linux-amd64.tar.gz  100%[===================>] 65M  20.0MB/s    in 3.2s"},
            {"cmd": "sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz", "output": ""},
            {"cmd": "echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc", "output": ""},
            {"cmd": "echo 'export GOPATH=$HOME/go' >> ~/.bashrc", "output": ""},
            {"cmd": "echo 'export PATH=$PATH:$GOPATH/bin' >> ~/.bashrc && source ~/.bashrc", "output": ""},
            {"cmd": "go version", "output": "go version go1.21.5 linux/amd64"},
            {"cmd": "mkdir -p $GOPATH/{bin,src,pkg}", "output": ""},
        ],
        "test_code": 'package main\nimport "fmt"\nfunc main() { fmt.Println("Hello, Go!") }',
        "test_cmd": "go run hello.go",
        "test_output": "Hello, Go!",
        "compile_cmd": "go build -o hello hello.go && ./hello",
    },
    "rust": {
        "name": "Rust",
        "description": "Set up Rust development environment",
        "steps": [
            {"cmd": "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y", "output": "info: downloading installer\ninfo: profile set to 'default'\ninfo: default toolchain set to 'stable-x86_64-unknown-linux-gnu'\nstable-x86_64-unknown-linux-gnu installed"},
            {"cmd": "source $HOME/.cargo/env", "output": ""},
            {"cmd": "rustc --version", "output": "rustc 1.74.1 (a28077b28 2023-12-04)"},
            {"cmd": "cargo --version", "output": "cargo 1.74.1 (ecb9851af 2023-10-18)"},
            {"cmd": "rustup component add rustfmt clippy", "output": "info: component 'rustfmt' is up to date\ninfo: component 'clippy' is up to date"},
        ],
        "test_code": 'fn main() { println!("Hello, Rust!"); }',
        "test_cmd": "rustc hello.rs && ./hello",
        "test_output": "Hello, Rust!",
        "compile_cmd": "cargo new hello_project && cd hello_project && cargo run",
    },
    "nodejs": {
        "name": "Node.js",
        "description": "Set up Node.js development environment",
        "steps": [
            {"cmd": "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -", "output": "## Installing the NodeSource Node.js 20.x repo..."},
            {"cmd": "sudo apt install -y nodejs", "output": "Setting up nodejs (20.10.0-1nodesource1) ..."},
            {"cmd": "node --version", "output": "v20.10.0"},
            {"cmd": "npm --version", "output": "10.2.3"},
            {"cmd": "sudo npm install -g yarn typescript ts-node", "output": "added 3 packages in 2s"},
            {"cmd": "yarn --version", "output": "1.22.21"},
            {"cmd": "tsc --version", "output": "Version 5.3.3"},
        ],
        "test_code": 'console.log("Hello, Node.js!");',
        "test_cmd": "node hello.js",
        "test_output": "Hello, Node.js!",
    },
    "python": {
        "name": "Python",
        "description": "Set up Python development environment with pyenv",
        "steps": [
            {"cmd": "sudo apt install -y python3 python3-pip python3-venv python3-dev", "output": "Setting up python3 (3.10.12-1~22.04) ..."},
            {"cmd": "curl https://pyenv.run | bash", "output": "Cloning into '/home/user/.pyenv'...\npyenv installed successfully"},
            {"cmd": "echo 'export PYENV_ROOT=\"$HOME/.pyenv\"' >> ~/.bashrc", "output": ""},
            {"cmd": "echo 'command -v pyenv >/dev/null || export PATH=\"$PYENV_ROOT/bin:$PATH\"' >> ~/.bashrc", "output": ""},
            {"cmd": "echo 'eval \"$(pyenv init -)\"' >> ~/.bashrc && source ~/.bashrc", "output": ""},
            {"cmd": "pyenv install 3.12.1", "output": "Downloading Python-3.12.1.tar.xz...\nInstalling Python-3.12.1..."},
            {"cmd": "pyenv global 3.12.1", "output": ""},
            {"cmd": "python --version", "output": "Python 3.12.1"},
            {"cmd": "pip install --upgrade pip setuptools wheel", "output": "Successfully installed pip-23.3.2 setuptools-69.0.3 wheel-0.42.0"},
        ],
        "test_code": 'print("Hello, Python!")',
        "test_cmd": "python hello.py",
        "test_output": "Hello, Python!",
    },
    "ruby": {
        "name": "Ruby",
        "description": "Set up Ruby development environment with rbenv",
        "steps": [
            {"cmd": "sudo apt install -y rbenv ruby-build", "output": "Setting up rbenv (1.2.0-1) ..."},
            {"cmd": "echo 'eval \"$(rbenv init -)\"' >> ~/.bashrc && source ~/.bashrc", "output": ""},
            {"cmd": "rbenv install 3.3.0", "output": "Downloading ruby-3.3.0.tar.gz...\nInstalling ruby-3.3.0..."},
            {"cmd": "rbenv global 3.3.0", "output": ""},
            {"cmd": "ruby --version", "output": "ruby 3.3.0 (2023-12-25 revision 5124f9ac75) [x86_64-linux]"},
            {"cmd": "gem install bundler rails", "output": "Successfully installed bundler-2.5.4\nSuccessfully installed rails-7.1.2"},
        ],
        "test_code": 'puts "Hello, Ruby!"',
        "test_cmd": "ruby hello.rb",
        "test_output": "Hello, Ruby!",
    },
    "php": {
        "name": "PHP",
        "description": "Set up PHP development environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y php php-cli php-fpm php-mysql php-xml php-curl php-mbstring php-zip composer", "output": "Setting up php8.1 (8.1.2-1ubuntu2.14) ..."},
            {"cmd": "php --version", "output": "PHP 8.1.2-1ubuntu2.14 (cli) (built: Aug 18 2023 11:41:11) (NTS)\nCopyright (c) The PHP Group"},
            {"cmd": "composer --version", "output": "Composer version 2.2.6 2022-02-04 17:00:38"},
        ],
        "test_code": '<?php echo "Hello, PHP!\\n"; ?>',
        "test_cmd": "php hello.php",
        "test_output": "Hello, PHP!",
    },
    "dotnet": {
        "name": ".NET (C#)",
        "description": "Set up .NET development environment",
        "steps": [
            {"cmd": "wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb", "output": "packages-microsoft-prod.deb  100%[===================>] 3.1K  --.-KB/s    in 0s"},
            {"cmd": "sudo dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb", "output": "Selecting previously unselected package packages-microsoft-prod."},
            {"cmd": "sudo apt update && sudo apt install -y dotnet-sdk-8.0", "output": "Setting up dotnet-sdk-8.0 (8.0.100-1) ..."},
            {"cmd": "dotnet --version", "output": "8.0.100"},
            {"cmd": "dotnet --list-sdks", "output": "8.0.100 [/usr/share/dotnet/sdk]"},
        ],
        "test_code": 'Console.WriteLine("Hello, C#!");',
        "test_cmd": "dotnet new console -o hello && cd hello && dotnet run",
        "test_output": "Hello, C#!",
    },
    "kotlin": {
        "name": "Kotlin",
        "description": "Set up Kotlin development environment",
        "steps": [
            {"cmd": "curl -s https://get.sdkman.io | bash", "output": "SDKMAN! installed successfully."},
            {"cmd": "source ~/.sdkman/bin/sdkman-init.sh", "output": ""},
            {"cmd": "sdk install kotlin", "output": "Downloading: kotlin 1.9.22\nInstalling: kotlin 1.9.22\nDone installing!"},
            {"cmd": "kotlin -version", "output": "Kotlin version 1.9.22-release-704 (JRE 17.0.9+9-Ubuntu-122.04)"},
        ],
        "test_code": 'fun main() { println("Hello, Kotlin!") }',
        "test_cmd": "kotlinc hello.kt -include-runtime -d hello.jar && java -jar hello.jar",
        "test_output": "Hello, Kotlin!",
    },
    "scala": {
        "name": "Scala",
        "description": "Set up Scala development environment",
        "steps": [
            {"cmd": "curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup -y", "output": "Installed scala, scalac, sbt, sbtn, ammonite, scala-cli"},
            {"cmd": "scala --version", "output": "Scala code runner version 3.3.1 -- Copyright 2002-2023, LAMP/EPFL"},
            {"cmd": "sbt --version", "output": "sbt version in this project: 1.9.8"},
        ],
        "test_code": 'object Hello extends App { println("Hello, Scala!") }',
        "test_cmd": "scala hello.scala",
        "test_output": "Hello, Scala!",
    },
    "haskell": {
        "name": "Haskell",
        "description": "Set up Haskell development environment",
        "steps": [
            {"cmd": "curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh", "output": "GHCup installed successfully"},
            {"cmd": "source ~/.ghcup/env", "output": ""},
            {"cmd": "ghcup install ghc 9.4.8", "output": "[ Info  ] downloading: https://downloads.haskell.org/~ghc/9.4.8/ghc-9.4.8-x86_64-deb10-linux.tar.xz"},
            {"cmd": "ghcup set ghc 9.4.8", "output": "[ Info  ] GHC 9.4.8 successfully set as default version"},
            {"cmd": "ghc --version", "output": "The Glorious Glasgow Haskell Compilation System, version 9.4.8"},
            {"cmd": "cabal --version", "output": "cabal-install version 3.10.2.1"},
        ],
        "test_code": 'main = putStrLn "Hello, Haskell!"',
        "test_cmd": "ghc hello.hs && ./hello",
        "test_output": "Hello, Haskell!",
    },
    "elixir": {
        "name": "Elixir",
        "description": "Set up Elixir development environment",
        "steps": [
            {"cmd": "sudo apt install -y erlang elixir", "output": "Setting up erlang (1:24.3.4.1+dfsg-1) ...\nSetting up elixir (1.12.2-1) ..."},
            {"cmd": "elixir --version", "output": "Erlang/OTP 24 [erts-12.3.2.1]\nElixir 1.12.2 (compiled with Erlang/OTP 24)"},
            {"cmd": "mix --version", "output": "Mix 1.12.2 (compiled with Erlang/OTP 24)"},
        ],
        "test_code": 'IO.puts "Hello, Elixir!"',
        "test_cmd": "elixir hello.exs",
        "test_output": "Hello, Elixir!",
    },
    "clojure": {
        "name": "Clojure",
        "description": "Set up Clojure development environment",
        "steps": [
            {"cmd": "curl -O https://download.clojure.org/install/linux-install-1.11.1.1435.sh", "output": ""},
            {"cmd": "chmod +x linux-install-1.11.1.1435.sh && sudo ./linux-install-1.11.1.1435.sh", "output": "Downloading and installing Clojure CLI tools"},
            {"cmd": "clj --version", "output": "Clojure CLI version 1.11.1.1435"},
        ],
        "test_code": '(println "Hello, Clojure!")',
        "test_cmd": "clj -e '(println \"Hello, Clojure!\")'",
        "test_output": "Hello, Clojure!",
    },
    "lua": {
        "name": "Lua",
        "description": "Set up Lua development environment",
        "steps": [
            {"cmd": "sudo apt install -y lua5.4 liblua5.4-dev luarocks", "output": "Setting up lua5.4 (5.4.4-3) ..."},
            {"cmd": "lua -v", "output": "Lua 5.4.4  Copyright (C) 1994-2022 Lua.org, PUC-Rio"},
            {"cmd": "luarocks --version", "output": "luarocks 3.8.0"},
        ],
        "test_code": 'print("Hello, Lua!")',
        "test_cmd": "lua hello.lua",
        "test_output": "Hello, Lua!",
    },
    "perl": {
        "name": "Perl",
        "description": "Set up Perl development environment",
        "steps": [
            {"cmd": "sudo apt install -y perl cpanminus", "output": "Setting up perl (5.34.0-3ubuntu1.3) ..."},
            {"cmd": "perl --version | head -2", "output": "This is perl 5, version 34, subversion 0 (v5.34.0)"},
            {"cmd": "cpanm --version", "output": "cpanm (App::cpanminus) version 1.7044"},
        ],
        "test_code": 'print "Hello, Perl!\\n";',
        "test_cmd": "perl hello.pl",
        "test_output": "Hello, Perl!",
    },
    "r": {
        "name": "R",
        "description": "Set up R development environment",
        "steps": [
            {"cmd": "sudo apt install -y r-base r-base-dev", "output": "Setting up r-base (4.1.2-1ubuntu2) ..."},
            {"cmd": "R --version | head -1", "output": "R version 4.1.2 (2021-11-01) -- \"Bird Hippie\""},
            {"cmd": "sudo Rscript -e 'install.packages(c(\"tidyverse\", \"ggplot2\"), repos=\"https://cran.rstudio.com/\")'", "output": "Installing packages into '/usr/local/lib/R/site-library'"},
        ],
        "test_code": 'print("Hello, R!")',
        "test_cmd": "Rscript hello.R",
        "test_output": '[1] "Hello, R!"',
    },
    "julia": {
        "name": "Julia",
        "description": "Set up Julia development environment",
        "steps": [
            {"cmd": "wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz", "output": "julia-1.10.0-linux-x86_64.tar.gz  100%[===================>] 130M  25.0MB/s    in 5.2s"},
            {"cmd": "tar xzf julia-1.10.0-linux-x86_64.tar.gz", "output": ""},
            {"cmd": "sudo mv julia-1.10.0 /opt/julia", "output": ""},
            {"cmd": "sudo ln -s /opt/julia/bin/julia /usr/local/bin/julia", "output": ""},
            {"cmd": "julia --version", "output": "julia version 1.10.0"},
        ],
        "test_code": 'println("Hello, Julia!")',
        "test_cmd": "julia hello.jl",
        "test_output": "Hello, Julia!",
    },
    "zig": {
        "name": "Zig",
        "description": "Set up Zig development environment",
        "steps": [
            {"cmd": "wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz", "output": "zig-linux-x86_64-0.11.0.tar.xz  100%[===================>] 45M  20.0MB/s    in 2.2s"},
            {"cmd": "tar xf zig-linux-x86_64-0.11.0.tar.xz", "output": ""},
            {"cmd": "sudo mv zig-linux-x86_64-0.11.0 /opt/zig", "output": ""},
            {"cmd": "sudo ln -s /opt/zig/zig /usr/local/bin/zig", "output": ""},
            {"cmd": "zig version", "output": "0.11.0"},
        ],
        "test_code": 'const std = @import("std");\npub fn main() void { std.debug.print("Hello, Zig!\\n", .{}); }',
        "test_cmd": "zig run hello.zig",
        "test_output": "Hello, Zig!",
    },
    "nim": {
        "name": "Nim",
        "description": "Set up Nim development environment",
        "steps": [
            {"cmd": "curl https://nim-lang.org/choosenim/init.sh -sSf | sh -s -- -y", "output": "choosenim-init: Downloading choosenim-0.8.4_linux_amd64\nchoosenim-init: Installed version 2.0.2"},
            {"cmd": "export PATH=$HOME/.nimble/bin:$PATH", "output": ""},
            {"cmd": "nim --version | head -1", "output": "Nim Compiler Version 2.0.2 [Linux: amd64]"},
        ],
        "test_code": 'echo "Hello, Nim!"',
        "test_cmd": "nim c -r hello.nim",
        "test_output": "Hello, Nim!",
    },
    "ocaml": {
        "name": "OCaml",
        "description": "Set up OCaml development environment",
        "steps": [
            {"cmd": "sudo apt install -y opam", "output": "Setting up opam (2.1.2-1) ..."},
            {"cmd": "opam init -y", "output": "Configuring from built-in defaults..."},
            {"cmd": "eval $(opam env)", "output": ""},
            {"cmd": "opam install -y dune utop", "output": "Done."},
            {"cmd": "ocaml --version", "output": "The OCaml toplevel, version 4.14.1"},
        ],
        "test_code": 'print_endline "Hello, OCaml!"',
        "test_cmd": "ocaml hello.ml",
        "test_output": "Hello, OCaml!",
    },
    "fortran": {
        "name": "Fortran",
        "description": "Set up Fortran development environment",
        "steps": [
            {"cmd": "sudo apt install -y gfortran", "output": "Setting up gfortran (4:11.2.0-1ubuntu1) ..."},
            {"cmd": "gfortran --version | head -1", "output": "GNU Fortran (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"},
        ],
        "test_code": 'program hello\n  print *, "Hello, Fortran!"\nend program hello',
        "test_cmd": "gfortran hello.f90 -o hello && ./hello",
        "test_output": " Hello, Fortran!",
    },
    "cobol": {
        "name": "COBOL",
        "description": "Set up COBOL development environment",
        "steps": [
            {"cmd": "sudo apt install -y gnucobol", "output": "Setting up gnucobol (3.1.2-1) ..."},
            {"cmd": "cobc --version | head -1", "output": "cobc (GnuCOBOL) 3.1.2.0"},
        ],
        "test_code": '       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO.\n       PROCEDURE DIVISION.\n           DISPLAY "Hello, COBOL!".\n           STOP RUN.',
        "test_cmd": "cobc -x hello.cob && ./hello",
        "test_output": "Hello, COBOL!",
    },
    "dart": {
        "name": "Dart",
        "description": "Set up Dart development environment",
        "steps": [
            {"cmd": "sudo apt install -y apt-transport-https", "output": ""},
            {"cmd": "wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/dart.gpg", "output": ""},
            {"cmd": "echo 'deb [signed-by=/usr/share/keyrings/dart.gpg arch=amd64] https://storage.googleapis.com/download.dartlang.org/linux/debian stable main' | sudo tee /etc/apt/sources.list.d/dart_stable.list", "output": ""},
            {"cmd": "sudo apt update && sudo apt install -y dart", "output": "Setting up dart (3.2.4-1) ..."},
            {"cmd": "dart --version", "output": "Dart SDK version: 3.2.4 (stable)"},
        ],
        "test_code": 'void main() { print("Hello, Dart!"); }',
        "test_cmd": "dart run hello.dart",
        "test_output": "Hello, Dart!",
    },
    "flutter": {
        "name": "Flutter",
        "description": "Set up Flutter development environment",
        "steps": [
            {"cmd": "sudo apt install -y clang cmake ninja-build pkg-config libgtk-3-dev liblzma-dev", "output": "Setting up dependencies..."},
            {"cmd": "git clone https://github.com/flutter/flutter.git -b stable ~/flutter", "output": "Cloning into '/home/user/flutter'..."},
            {"cmd": "echo 'export PATH=$PATH:$HOME/flutter/bin' >> ~/.bashrc && source ~/.bashrc", "output": ""},
            {"cmd": "flutter doctor", "output": "Doctor summary (to see all details, run flutter doctor -v):\n[✓] Flutter (Channel stable, 3.16.5)"},
            {"cmd": "flutter --version", "output": "Flutter 3.16.5 • channel stable"},
        ],
        "test_code": "flutter create hello_app",
        "test_cmd": "cd hello_app && flutter run -d linux",
        "test_output": "Running on Linux...",
    },
}

# =============================================================================
# DESKTOP ENVIRONMENT SETUP
# =============================================================================

DESKTOP_SETUPS = [
    {
        "task": "Install XFCE desktop environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y xfce4 xfce4-goodies", "output": "Setting up xfce4 (4.16) ...\nSetting up xfce4-goodies (4.16.0) ..."},
            {"cmd": "sudo apt install -y lightdm", "output": "Setting up lightdm (1.30.0-0ubuntu4) ..."},
            {"cmd": "sudo systemctl enable lightdm", "output": "Created symlink /etc/systemd/system/display-manager.service → /lib/systemd/system/lightdm.service."},
            {"cmd": "echo 'exec startxfce4' > ~/.xinitrc", "output": ""},
        ]
    },
    {
        "task": "Install GNOME desktop environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y ubuntu-desktop", "output": "Setting up ubuntu-desktop (1.481) ..."},
            {"cmd": "sudo systemctl set-default graphical.target", "output": "Created symlink /etc/systemd/system/default.target → /lib/systemd/system/graphical.target."},
        ]
    },
    {
        "task": "Install KDE Plasma desktop environment",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y kde-plasma-desktop sddm", "output": "Setting up kde-plasma-desktop (5:5.24.7-0ubuntu0.1) ..."},
            {"cmd": "sudo systemctl enable sddm", "output": "Created symlink /etc/systemd/system/display-manager.service → /lib/systemd/system/sddm.service."},
        ]
    },
    {
        "task": "Install i3 window manager",
        "steps": [
            {"cmd": "sudo apt install -y i3 i3status dmenu i3lock xterm", "output": "Setting up i3 (4.20.1-1) ..."},
            {"cmd": "echo 'exec i3' > ~/.xinitrc", "output": ""},
            {"cmd": "mkdir -p ~/.config/i3 && cp /etc/i3/config ~/.config/i3/config", "output": ""},
        ]
    },
    {
        "task": "Set up VNC server for remote desktop",
        "steps": [
            {"cmd": "sudo apt install -y tigervnc-standalone-server tigervnc-common", "output": "Setting up tigervnc-standalone-server (1.12.0+dfsg-4) ..."},
            {"cmd": "vncpasswd", "output": "Password:\nVerify:\nWould you like to enter a view-only password (y/n)? n"},
            {"cmd": "vncserver :1 -geometry 1920x1080 -depth 24", "output": "New 'hostname:1 (user)' desktop at :1 on machine hostname"},
            {"cmd": "vncserver -list", "output": "TigerVNC server sessions:\n\nX DISPLAY #\tRFB PORT #\tPROCESS ID\n:1\t\t5901\t\t12345"},
        ]
    },
    {
        "task": "Set up X11 forwarding over SSH",
        "steps": [
            {"cmd": "sudo apt install -y xauth x11-apps", "output": "Setting up xauth (1:1.1-1build2) ..."},
            {"cmd": "sudo sed -i 's/#X11Forwarding no/X11Forwarding yes/' /etc/ssh/sshd_config", "output": ""},
            {"cmd": "sudo sed -i 's/#X11DisplayOffset 10/X11DisplayOffset 10/' /etc/ssh/sshd_config", "output": ""},
            {"cmd": "sudo systemctl restart sshd", "output": ""},
            {"cmd": "ssh -X user@remote xclock", "output": "(displays clock window)"},
        ]
    },
]

# =============================================================================
# SSH AND REMOTE ACCESS
# =============================================================================

SSH_SCENARIOS = [
    {
        "task": "Set up SSH server",
        "steps": [
            {"cmd": "sudo apt install -y openssh-server", "output": "Setting up openssh-server (1:8.9p1-3ubuntu0.4) ..."},
            {"cmd": "sudo systemctl enable ssh", "output": "Synchronizing state of ssh.service..."},
            {"cmd": "sudo systemctl start ssh", "output": ""},
            {"cmd": "sudo systemctl status ssh", "output": "● ssh.service - OpenBSD Secure Shell server\n     Loaded: loaded (/lib/systemd/system/ssh.service; enabled)\n     Active: active (running)"},
            {"cmd": "sudo ufw allow ssh", "output": "Rule added\nRule added (v6)"},
        ]
    },
    {
        "task": "Generate SSH key pair",
        "steps": [
            {"cmd": "ssh-keygen -t ed25519 -C 'user@example.com' -f ~/.ssh/id_ed25519 -N ''", "output": "Generating public/private ed25519 key pair.\nYour identification has been saved in /home/user/.ssh/id_ed25519\nYour public key has been saved in /home/user/.ssh/id_ed25519.pub"},
            {"cmd": "chmod 600 ~/.ssh/id_ed25519", "output": ""},
            {"cmd": "chmod 644 ~/.ssh/id_ed25519.pub", "output": ""},
            {"cmd": "cat ~/.ssh/id_ed25519.pub", "output": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGx... user@example.com"},
        ]
    },
    {
        "task": "Copy SSH key to remote server",
        "steps": [
            {"cmd": "ssh-copy-id -i ~/.ssh/id_ed25519.pub user@192.168.1.100", "output": "/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed\nNumber of key(s) added: 1"},
            {"cmd": "ssh user@192.168.1.100 'echo Connection successful'", "output": "Connection successful"},
        ]
    },
    {
        "task": "Set up SSH config for easy connections",
        "steps": [
            {"cmd": "cat > ~/.ssh/config << 'EOF'\nHost myserver\n    HostName 192.168.1.100\n    User admin\n    IdentityFile ~/.ssh/id_ed25519\n    Port 22\n    ServerAliveInterval 60\nEOF", "output": ""},
            {"cmd": "chmod 600 ~/.ssh/config", "output": ""},
            {"cmd": "ssh myserver 'hostname'", "output": "myserver"},
        ]
    },
    {
        "task": "Set up SSH port forwarding (tunnel)",
        "steps": [
            {"cmd": "ssh -L 8080:localhost:80 user@remote-server -N -f", "output": ""},
            {"cmd": "curl http://localhost:8080", "output": "<!DOCTYPE html>..."},
            {"cmd": "ssh -R 9090:localhost:3000 user@remote-server -N -f", "output": ""},
        ]
    },
    {
        "task": "Set up SSH jump host (bastion)",
        "steps": [
            {"cmd": "cat >> ~/.ssh/config << 'EOF'\nHost bastion\n    HostName bastion.example.com\n    User admin\n\nHost internal-server\n    HostName 10.0.0.50\n    User admin\n    ProxyJump bastion\nEOF", "output": ""},
            {"cmd": "ssh internal-server 'hostname'", "output": "internal-server"},
        ]
    },
    {
        "task": "Secure SSH server configuration",
        "steps": [
            {"cmd": "sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup", "output": ""},
            {"cmd": "sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config", "output": ""},
            {"cmd": "sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config", "output": ""},
            {"cmd": "sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config", "output": ""},
            {"cmd": "sudo sshd -t", "output": ""},
            {"cmd": "sudo systemctl restart sshd", "output": ""},
        ]
    },
]

# =============================================================================
# DOCKER AND CONTAINERIZATION
# =============================================================================

DOCKER_SCENARIOS = [
    {
        "task": "Install Docker",
        "steps": [
            {"cmd": "sudo apt update", "output": "Reading package lists... Done"},
            {"cmd": "sudo apt install -y ca-certificates curl gnupg", "output": ""},
            {"cmd": "sudo install -m 0755 -d /etc/apt/keyrings", "output": ""},
            {"cmd": "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg", "output": ""},
            {"cmd": "sudo chmod a+r /etc/apt/keyrings/docker.gpg", "output": ""},
            {"cmd": 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null', "output": ""},
            {"cmd": "sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin", "output": "Setting up docker-ce (5:24.0.7-1~ubuntu.22.04~jammy) ..."},
            {"cmd": "sudo usermod -aG docker $USER", "output": ""},
            {"cmd": "docker --version", "output": "Docker version 24.0.7, build afdd53b"},
        ]
    },
    {
        "task": "Run a Docker container",
        "steps": [
            {"cmd": "docker pull nginx:latest", "output": "latest: Pulling from library/nginx\nDigest: sha256:...\nStatus: Downloaded newer image for nginx:latest"},
            {"cmd": "docker run -d -p 8080:80 --name my-nginx nginx:latest", "output": "a1b2c3d4e5f6..."},
            {"cmd": "docker ps", "output": "CONTAINER ID   IMAGE          COMMAND                  CREATED         STATUS         PORTS                  NAMES\na1b2c3d4e5f6   nginx:latest   \"/docker-entrypoint.…\"   5 seconds ago   Up 4 seconds   0.0.0.0:8080->80/tcp   my-nginx"},
            {"cmd": "curl http://localhost:8080", "output": "<!DOCTYPE html>\n<html>\n<head>\n<title>Welcome to nginx!</title>..."},
        ]
    },
    {
        "task": "Build a Docker image",
        "steps": [
            {"cmd": "cat > Dockerfile << 'EOF'\nFROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"app.py\"]\nEOF", "output": ""},
            {"cmd": "docker build -t my-python-app:latest .", "output": "Step 1/6 : FROM python:3.11-slim\n ---> abc123\nStep 6/6 : CMD [\"python\", \"app.py\"]\n ---> Running in def456\nSuccessfully built ghi789\nSuccessfully tagged my-python-app:latest"},
            {"cmd": "docker images | grep my-python-app", "output": "my-python-app   latest    ghi789    10 seconds ago   150MB"},
        ]
    },
    {
        "task": "Use Docker Compose",
        "steps": [
            {"cmd": "cat > docker-compose.yml << 'EOF'\nversion: '3.8'\nservices:\n  web:\n    build: .\n    ports:\n      - \"5000:5000\"\n  redis:\n    image: redis:alpine\nEOF", "output": ""},
            {"cmd": "docker compose up -d", "output": "Creating network \"app_default\" with the default driver\nCreating app_redis_1 ... done\nCreating app_web_1   ... done"},
            {"cmd": "docker compose ps", "output": "NAME                COMMAND                  SERVICE             STATUS              PORTS\napp-redis-1         \"docker-entrypoint.s…\"   redis               running             6379/tcp\napp-web-1           \"python app.py\"          web                 running             0.0.0.0:5000->5000/tcp"},
            {"cmd": "docker compose logs web", "output": "web_1  | * Running on http://0.0.0.0:5000"},
            {"cmd": "docker compose down", "output": "Stopping app_web_1   ... done\nStopping app_redis_1 ... done\nRemoving app_web_1   ... done\nRemoving app_redis_1 ... done"},
        ]
    },
]

# =============================================================================
# DATABASE SETUP
# =============================================================================

DATABASE_SETUPS = [
    {
        "task": "Install and configure PostgreSQL",
        "steps": [
            {"cmd": "sudo apt install -y postgresql postgresql-contrib", "output": "Setting up postgresql-14 (14.10-0ubuntu0.22.04.1) ..."},
            {"cmd": "sudo systemctl start postgresql", "output": ""},
            {"cmd": "sudo systemctl enable postgresql", "output": ""},
            {"cmd": "sudo -u postgres psql -c \"CREATE USER myuser WITH PASSWORD 'mypassword';\"", "output": "CREATE ROLE"},
            {"cmd": "sudo -u postgres psql -c \"CREATE DATABASE mydb OWNER myuser;\"", "output": "CREATE DATABASE"},
            {"cmd": "sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;\"", "output": "GRANT"},
            {"cmd": "psql -h localhost -U myuser -d mydb -c 'SELECT version();'", "output": "PostgreSQL 14.10 (Ubuntu 14.10-0ubuntu0.22.04.1)"},
        ]
    },
    {
        "task": "Install and configure MySQL",
        "steps": [
            {"cmd": "sudo apt install -y mysql-server", "output": "Setting up mysql-server-8.0 (8.0.35-0ubuntu0.22.04.1) ..."},
            {"cmd": "sudo systemctl start mysql", "output": ""},
            {"cmd": "sudo mysql_secure_installation", "output": "Securing the MySQL server deployment..."},
            {"cmd": "sudo mysql -e \"CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'mypassword';\"", "output": ""},
            {"cmd": "sudo mysql -e \"CREATE DATABASE mydb;\"", "output": ""},
            {"cmd": "sudo mysql -e \"GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'localhost';\"", "output": ""},
            {"cmd": "mysql -u myuser -p mydb -e 'SELECT VERSION();'", "output": "8.0.35-0ubuntu0.22.04.1"},
        ]
    },
    {
        "task": "Install and configure MongoDB",
        "steps": [
            {"cmd": "curl -fsSL https://pgp.mongodb.com/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor", "output": ""},
            {"cmd": "echo 'deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse' | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list", "output": ""},
            {"cmd": "sudo apt update && sudo apt install -y mongodb-org", "output": "Setting up mongodb-org (7.0.4) ..."},
            {"cmd": "sudo systemctl start mongod", "output": ""},
            {"cmd": "sudo systemctl enable mongod", "output": ""},
            {"cmd": "mongosh --eval 'db.version()'", "output": "7.0.4"},
        ]
    },
    {
        "task": "Install and configure Redis",
        "steps": [
            {"cmd": "sudo apt install -y redis-server", "output": "Setting up redis-server (5:6.0.16-1ubuntu1) ..."},
            {"cmd": "sudo systemctl start redis-server", "output": ""},
            {"cmd": "sudo systemctl enable redis-server", "output": ""},
            {"cmd": "redis-cli ping", "output": "PONG"},
            {"cmd": "redis-cli INFO server | head -5", "output": "# Server\nredis_version:6.0.16\nredis_git_sha1:00000000"},
        ]
    },
]

# =============================================================================
# WEB SERVERS
# =============================================================================

WEBSERVER_SETUPS = [
    {
        "task": "Install and configure Nginx",
        "steps": [
            {"cmd": "sudo apt install -y nginx", "output": "Setting up nginx (1.18.0-6ubuntu14.4) ..."},
            {"cmd": "sudo systemctl start nginx", "output": ""},
            {"cmd": "sudo systemctl enable nginx", "output": ""},
            {"cmd": "sudo ufw allow 'Nginx Full'", "output": "Rule added"},
            {"cmd": "curl http://localhost", "output": "<!DOCTYPE html>\n<html>\n<head>\n<title>Welcome to nginx!</title>"},
            {"cmd": "sudo nginx -t", "output": "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\nnginx: configuration file /etc/nginx/nginx.conf test is successful"},
        ]
    },
    {
        "task": "Install and configure Apache",
        "steps": [
            {"cmd": "sudo apt install -y apache2", "output": "Setting up apache2 (2.4.52-1ubuntu4.7) ..."},
            {"cmd": "sudo systemctl start apache2", "output": ""},
            {"cmd": "sudo systemctl enable apache2", "output": ""},
            {"cmd": "sudo ufw allow 'Apache Full'", "output": "Rule added"},
            {"cmd": "curl http://localhost", "output": "<!DOCTYPE html PUBLIC...Apache2 Ubuntu Default Page"},
            {"cmd": "sudo apache2ctl configtest", "output": "Syntax OK"},
        ]
    },
    {
        "task": "Set up Nginx reverse proxy",
        "steps": [
            {"cmd": "sudo cat > /etc/nginx/sites-available/myapp << 'EOF'\nserver {\n    listen 80;\n    server_name myapp.example.com;\n\n    location / {\n        proxy_pass http://localhost:3000;\n        proxy_http_version 1.1;\n        proxy_set_header Upgrade $http_upgrade;\n        proxy_set_header Connection 'upgrade';\n        proxy_set_header Host $host;\n        proxy_cache_bypass $http_upgrade;\n    }\n}\nEOF", "output": ""},
            {"cmd": "sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/", "output": ""},
            {"cmd": "sudo nginx -t", "output": "nginx: configuration file /etc/nginx/nginx.conf test is successful"},
            {"cmd": "sudo systemctl reload nginx", "output": ""},
        ]
    },
    {
        "task": "Set up SSL with Let's Encrypt",
        "steps": [
            {"cmd": "sudo apt install -y certbot python3-certbot-nginx", "output": "Setting up certbot (1.21.0-1) ..."},
            {"cmd": "sudo certbot --nginx -d example.com -d www.example.com --non-interactive --agree-tos -m admin@example.com", "output": "Congratulations! Your certificate and chain have been saved at:\n/etc/letsencrypt/live/example.com/fullchain.pem"},
            {"cmd": "sudo certbot renew --dry-run", "output": "Congratulations, all simulated renewals succeeded"},
        ]
    },
]

# =============================================================================
# FILE DOWNLOADS AND MANAGEMENT
# =============================================================================

DOWNLOAD_SCENARIOS = [
    {
        "task": "Download file with wget",
        "steps": [
            {"cmd": "wget https://example.com/file.zip", "output": "file.zip  100%[===================>] 10.5M  5.00MB/s    in 2.1s"},
            {"cmd": "wget -O custom_name.zip https://example.com/file.zip", "output": "custom_name.zip  100%[===================>] 10.5M  5.00MB/s    in 2.1s"},
            {"cmd": "wget -c https://example.com/large_file.zip", "output": "Continuing at byte position 52428800.\nlarge_file.zip  100%[+++++++============>] 100M  10.0MB/s    in 5.0s"},
            {"cmd": "wget -r -np -nd https://example.com/files/", "output": "Downloaded: 15 files, 50M in 10s (5.0 MB/s)"},
        ]
    },
    {
        "task": "Download file with curl",
        "steps": [
            {"cmd": "curl -O https://example.com/file.zip", "output": "  % Total    % Received % Xferd  Average Speed\n100 10.5M  100 10.5M    0     0  5.0M      0  0:00:02  0:00:02 --:--:-- 5.0M"},
            {"cmd": "curl -o custom_name.zip https://example.com/file.zip", "output": "  % Total    % Received % Xferd  Average Speed\n100 10.5M  100 10.5M    0     0  5.0M      0  0:00:02  0:00:02 --:--:-- 5.0M"},
            {"cmd": "curl -C - -O https://example.com/large_file.zip", "output": "** Resuming transfer from byte position 52428800"},
            {"cmd": "curl -L -O https://example.com/redirect_file.zip", "output": "Following redirect to https://cdn.example.com/file.zip"},
        ]
    },
    {
        "task": "Download with aria2 (parallel downloads)",
        "steps": [
            {"cmd": "sudo apt install -y aria2", "output": "Setting up aria2 (1.36.0-1) ..."},
            {"cmd": "aria2c -x 16 -s 16 https://example.com/large_file.zip", "output": "[#abc123 50MiB/100MiB(50%) CN:16 DL:50MiB]\nDownload complete: large_file.zip"},
            {"cmd": "aria2c -i urls.txt", "output": "Download Results:\ngid   |stat|avg speed  |path/URI\n======+====+===========+=========\nabc123|OK  |    10MiB/s|./file1.zip\ndef456|OK  |    10MiB/s|./file2.zip"},
        ]
    },
    {
        "task": "Clone Git repository",
        "steps": [
            {"cmd": "git clone https://github.com/user/repo.git", "output": "Cloning into 'repo'...\nremote: Enumerating objects: 1000, done.\nremote: Total 1000 (delta 0), reused 0 (delta 0)\nReceiving objects: 100% (1000/1000), 5.00 MiB | 10.00 MiB/s, done."},
            {"cmd": "git clone --depth 1 https://github.com/user/repo.git", "output": "Cloning into 'repo'...\nReceiving objects: 100% (100/100), 500.00 KiB | 5.00 MiB/s, done."},
            {"cmd": "git clone --branch develop https://github.com/user/repo.git", "output": "Cloning into 'repo'...\nSwitched to branch 'develop'"},
        ]
    },
    {
        "task": "Extract archives",
        "steps": [
            {"cmd": "tar -xzf archive.tar.gz", "output": ""},
            {"cmd": "tar -xjf archive.tar.bz2", "output": ""},
            {"cmd": "tar -xf archive.tar.xz", "output": ""},
            {"cmd": "unzip archive.zip -d output_dir", "output": "Archive:  archive.zip\n  inflating: output_dir/file1.txt\n  inflating: output_dir/file2.txt"},
            {"cmd": "7z x archive.7z", "output": "Everything is Ok\nFiles: 10\nSize: 50000000"},
            {"cmd": "gunzip file.gz", "output": ""},
        ]
    },
]

# =============================================================================
# SYSTEM MONITORING
# =============================================================================

MONITORING_SCENARIOS = [
    {
        "task": "Monitor system resources",
        "steps": [
            {"cmd": "htop", "output": "(interactive process viewer)"},
            {"cmd": "top -bn1 | head -20", "output": "top - 10:00:00 up 10 days,  5:30,  2 users,  load average: 0.50, 0.40, 0.30\nTasks: 200 total,   1 running, 199 sleeping,   0 stopped,   0 zombie\n%Cpu(s):  5.0 us,  2.0 sy,  0.0 ni, 92.0 id,  1.0 wa,  0.0 hi,  0.0 si,  0.0 st"},
            {"cmd": "free -h", "output": "              total        used        free      shared  buff/cache   available\nMem:           16Gi       4.0Gi       8.0Gi       500Mi       4.0Gi        11Gi\nSwap:          2.0Gi          0B       2.0Gi"},
            {"cmd": "df -h", "output": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       100G   45G   55G  45% /\n/dev/sdb1       500G  200G  300G  40% /data"},
            {"cmd": "iostat -x 1 3", "output": "avg-cpu:  %user   %nice %system %iowait  %steal   %idle\n           5.00    0.00    2.00    1.00    0.00   92.00"},
            {"cmd": "vmstat 1 5", "output": "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st"},
        ]
    },
    {
        "task": "Monitor network traffic",
        "steps": [
            {"cmd": "sudo apt install -y iftop nethogs", "output": "Setting up iftop (1.0~pre4-7) ..."},
            {"cmd": "sudo iftop -i eth0", "output": "(interactive network monitor)"},
            {"cmd": "sudo nethogs eth0", "output": "(per-process network usage)"},
            {"cmd": "netstat -tuln", "output": "Active Internet connections (only servers)\nProto Recv-Q Send-Q Local Address           Foreign Address         State\ntcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN\ntcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN"},
            {"cmd": "ss -tuln", "output": "Netid  State   Recv-Q  Send-Q   Local Address:Port   Peer Address:Port\ntcp    LISTEN  0       128      0.0.0.0:22            0.0.0.0:*\ntcp    LISTEN  0       511      0.0.0.0:80            0.0.0.0:*"},
        ]
    },
    {
        "task": "Check system logs",
        "steps": [
            {"cmd": "sudo journalctl -xe", "output": "-- Logs begin at Mon 2024-01-01 00:00:00 UTC, end at Wed 2024-01-24 10:00:00 UTC. --"},
            {"cmd": "sudo journalctl -u nginx --since '1 hour ago'", "output": "Jan 24 09:00:00 hostname nginx[1234]: 192.168.1.1 - - [24/Jan/2024:09:00:00 +0000] \"GET / HTTP/1.1\" 200"},
            {"cmd": "sudo tail -f /var/log/syslog", "output": "Jan 24 10:00:00 hostname systemd[1]: Started Session 100 of user admin."},
            {"cmd": "sudo dmesg | tail -20", "output": "[12345.678901] eth0: link up (1000Mbps/Full duplex)"},
        ]
    },
]


def generate_apt_install_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate an apt package installation sample."""
    scenario = random.choice(APT_PACKAGES)
    return _generate_multi_step_sample(t, scenario, "apt_install")


def generate_language_setup_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a programming language setup sample."""
    lang_key = random.choice(list(LANGUAGE_SETUPS.keys()))
    lang = LANGUAGE_SETUPS[lang_key]
    
    steps_text = ""
    for step in lang["steps"]:
        steps_text += f"{t['exec_start']}\n$ {step['cmd']}\n{t['exec_end']}\n"
        if step.get("output"):
            steps_text += f"{t['exec_result']}\n{step['output']}\n{t['exec_result_end']}\n"
        else:
            steps_text += f"{t['exec_result']}(no output){t['exec_result_end']}\n"
    
    # Add test code execution
    if "test_code" in lang and "test_cmd" in lang:
        steps_text += f"\nLet me verify the installation by running a test:\n\n"
        steps_text += f"{t['code_start']}{t['lang_python'] if 'python' in lang_key else t['code_start']}\n{lang['test_code']}\n{t['code_end']}\n\n"
        steps_text += f"{t['exec_start']}\n$ {lang['test_cmd']}\n{t['exec_end']}\n"
        steps_text += f"{t['exec_result']}\n{lang['test_output']}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{lang['description']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you set up {lang['name']}. Let me install and configure it:\n\n"
        f"{steps_text}"
        f"\n{lang['name']} is now installed and ready to use!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "language_setup", "language": lang_key}


def generate_desktop_setup_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a desktop environment setup sample."""
    scenario = random.choice(DESKTOP_SETUPS)
    return _generate_multi_step_sample(t, scenario, "desktop_setup")


def generate_ssh_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate an SSH configuration sample."""
    scenario = random.choice(SSH_SCENARIOS)
    return _generate_multi_step_sample(t, scenario, "ssh_setup")


def generate_docker_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a Docker usage sample."""
    scenario = random.choice(DOCKER_SCENARIOS)
    return _generate_multi_step_sample(t, scenario, "docker")


def generate_database_setup_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a database setup sample."""
    scenario = random.choice(DATABASE_SETUPS)
    return _generate_multi_step_sample(t, scenario, "database_setup")


def generate_webserver_setup_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a web server setup sample."""
    scenario = random.choice(WEBSERVER_SETUPS)
    return _generate_multi_step_sample(t, scenario, "webserver_setup")


def generate_download_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a file download sample."""
    scenario = random.choice(DOWNLOAD_SCENARIOS)
    return _generate_multi_step_sample(t, scenario, "download")


def generate_monitoring_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a system monitoring sample."""
    scenario = random.choice(MONITORING_SCENARIOS)
    return _generate_multi_step_sample(t, scenario, "monitoring")


def _generate_multi_step_sample(t: Dict[str, str], scenario: Dict, sample_type: str) -> Dict[str, Any]:
    """Helper to generate multi-step execution samples."""
    steps_text = ""
    for step in scenario["steps"]:
        steps_text += f"{t['exec_start']}\n$ {step['cmd']}\n{t['exec_end']}\n"
        if "error" in step:
            steps_text += f"{t['exec_error']}\n{step['error']}\n{t['exec_error_end']}\n"
        elif step.get("output"):
            steps_text += f"{t['exec_result']}\n{step['output']}\n{t['exec_result_end']}\n"
        else:
            steps_text += f"{t['exec_result']}(no output){t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{scenario['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {scenario['task'].lower()}. Let me execute the necessary commands:\n\n"
        f"{steps_text}"
        f"\nDone! {scenario['task']} completed successfully.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": sample_type}


def generate_dataset(output_dir: str, samples_per_type: int = 2000):
    """Generate all system administration datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    t = SPECIAL_TOKENS
    
    generators = [
        ("apt_install_dataset.jsonl", generate_apt_install_sample, "APT Package Installation"),
        ("language_setup_dataset.jsonl", generate_language_setup_sample, "Language Environment Setup"),
        ("desktop_setup_dataset.jsonl", generate_desktop_setup_sample, "Desktop Environment Setup"),
        ("ssh_setup_dataset.jsonl", generate_ssh_sample, "SSH Configuration"),
        ("docker_dataset.jsonl", generate_docker_sample, "Docker Usage"),
        ("database_setup_dataset.jsonl", generate_database_setup_sample, "Database Setup"),
        ("webserver_setup_dataset.jsonl", generate_webserver_setup_sample, "Web Server Setup"),
        ("download_dataset.jsonl", generate_download_sample, "File Downloads"),
        ("monitoring_dataset.jsonl", generate_monitoring_sample, "System Monitoring"),
    ]
    
    for filename, generator, desc in generators:
        filepath = os.path.join(output_dir, filename)
        print(f"📝 Generating {desc} dataset ({samples_per_type} samples)...")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(samples_per_type):
                sample = generator(t)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"   ✅ Saved to {filepath}")
    
    print(f"\n✅ Generated {len(generators)} datasets with {samples_per_type} samples each")
    print(f"   Total: {len(generators) * samples_per_type} samples")


# Removed standalone execution - use as module instead
# # if __name__ == "__main__":
#     output_dir = os.path.join(os.path.dirname(__file__), "data")
    # generate_dataset(output_dir, samples_per_type=2000)
