#!/bin/bash

# Sprawdzenie czy podano katalog jako argument
if [ "$#" -lt 1 ]; then
  echo "Użycie: $0 katalog1 [katalog2 ...]"
  exit 1
fi

# Funkcja przetwarzająca katalog
process_directory() {
  local dir=$1
  local wip_dir="$dir/wip"

  # Sprawdzenie, czy katalog wip istnieje, jeśli nie, to go tworzymy
  if [ ! -d "$wip_dir" ]; then
    echo "Katalog $wip_dir nie istnieje, tworzenie katalogu..."
    mkdir -p "$wip_dir"
  fi

  # Znajdowanie plików *.cut.png w katalogu wip i sortowanie według naturalnych liczb (sort -V)
  mapfile -t cut_files < <(find "$wip_dir" -name "*.cut.png" | sort -V)

  # Filtrowanie i ignorowanie plików zaczynających się od "Aligned"
  cut_files=("${cut_files[@]/*Aligned*/}")

  # Sprawdzenie, czy znaleziono jakieś pliki
  if [ "${#cut_files[@]}" -eq 0 ]; then
    echo "Nie znaleziono plików *.cut.png w katalogu $wip_dir"
    return
  fi

  #echo "Znalezione pliki: ${cut_files[@]}"  # Debugowanie: wyświetlenie znalezionych plików

  # Przetwarzanie metod od -a1 do -a6
  for i in {11..11}; do
    method="-a$i"

    # Ustawienie pliku początkowego dla tej metody
    previous_file=""

    # Przetwarzanie wszystkich plików dla danej metody
    for file in "${cut_files[@]}"; do
      # Sprawdzamy, czy zmienna $file nie jest pusta
      if [ -z "$file" ]; then
        #echo "Błąd: Zmienna 'file' jest pusta."
        continue  # Pomijamy ten krok
      fi

      # Debugowanie: wyświetlenie aktualnego pliku
      #echo "Aktualny plik: $file"

      # Oryginalna nazwa pliku
      base_name=$(basename "$file")

      # Jeśli plik z poprzedniego kroku istnieje, używamy go jako pierwszy argument
      if [ -n "$previous_file" ] && [ -f "$previous_file" ]; then
        first_file="$previous_file"
      else
        first_file="$file"  # Jeśli nie ma poprzedniego, zaczynamy od tego samego pliku
      fi

      # Tworzenie komendy align jako zmiennej
      command="./align $method $first_file $file"

      # Debugowanie: wyświetlenie komendy
      #echo "Wykonuję komendę: $command"

      # Wykonywanie komendy
      align_output=$($command)

      # Usuwanie nowych linii z wyjścia komendy
      #align_output=$(echo "$align_output" | tr -d '\n')

      # Nazwa pliku wyjściowego (przwidywana)
      aligned_file="$wip_dir/Aligned-a$i.$base_name"  # Zachowujemy oryginalną nazwę

      # Sprawdzanie, czy plik wyjściowy został stworzony
      if [ -f "$aligned_file" ]; then
        previous_file="$aligned_file"  # Ustawienie obecnego pliku jako poprzedni na następny krok
      else
        echo "Błąd: Plik $aligned_file nie został stworzony."
      fi

      # Tworzenie nazwy pliku logu w katalogu pliku wyjściowego
      log_file="${aligned_file}.txt"

      # Zapisanie komendy i wyjścia do pliku logu
      echo -e "$command\t$align_output" > "$log_file"
      echo -e "$command\t$align_output"
    done
  done
}

# Przetwarzanie każdego katalogu podanego jako argument
for directory in "$@"; do
  if [ -d "$directory" ]; then
    process_directory "$directory"
  else
    echo "Katalog $directory nie istnieje"
  fi
done
