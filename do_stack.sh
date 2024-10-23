#!/bin/bash

# Sprawdzenie, czy podano katalog jako argument
if [ "$#" -lt 1 ]; then
  echo "Użycie: $0 katalog1 [katalog2 ...]"
  exit 1
fi

# Funkcja przetwarzająca katalog
process_directory() {
  local dir=$1
  local wip_dir="$dir/wip"

  # Sprawdzenie, czy katalog istnieje
  if [ ! -d "$dir" ]; then
    echo "Katalog $dir nie istnieje."
    return
  fi

  # Sprawdzenie, czy katalog wip istnieje, jeśli nie, to go tworzymy
  if [ ! -d "$wip_dir" ]; then
    echo "Katalog $wip_dir nie istnieje, tworzenie katalogu..."
    mkdir -p "$wip_dir"
  fi

  # Przetwarzanie metod -m1 do -m9 i -a4 oraz -a8
  for m in {7..8}; do
    for a in {4..4}; do
      method="-m$m -a$a"

      # Tworzenie komendy stack
      command="./stack $method $dir"

      # Debugowanie: wyświetlenie komendy
      echo "Wykonuję komendę: $command"

      # Wykonywanie komendy i zapisanie wyjścia
      stack_output=$($command)

      # Usuwanie nowych linii z wyjścia komendy
      #stack_output=$(echo "$stack_output" | tr -d '\n')

      # Tworzenie nazwy pliku logu w katalogu wip
      #log_file="$wip_dir/stack-m$m-a$a-log.txt"

      # Zapisanie komendy i wyjścia do pliku logu
      #echo -e "$command\t$stack_output" > "$log_file"
      #echo -e "$command\t$stack_output"
    done
  done
}

# Przetwarzanie każdego katalogu podanego jako argument
for directory in "$@"; do
  process_directory "$directory"
done
