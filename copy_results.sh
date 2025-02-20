 #!/bin/bash

# Ścieżka do katalogu "data"
base_dir="data"

# Przechodzimy przez wszystkie katalogi w ścieżce "data/*/*/wip/"
for dir in $base_dir/*/*/wip; do
  if [ -d "$dir" ]; then
    # Określamy katalog docelowy "results" obok "wip"
    result_dir="$(dirname "$dir")/results"
    
    # Tworzymy katalog "results", jeśli nie istnieje
    mkdir -p "$result_dir"
    
    # Pliki do skopiowania
    files=("Stack-m11.Aligned-a12_edge_1.png" 
           "Stack-m11.Aligned-a12_edge_2.png"
           "Stack-m11.Aligned-a12_edge_1.txt"
           "Stack-m11.Aligned-a12_edge_2.txt")
    
    # Przechodzimy przez listę plików i kopiujemy je
    for file in "${files[@]}"; do
      src_file="$dir/$file"
      if [ -f "$src_file" ]; then
        cp "$src_file" "$result_dir"
        echo "Skopiowano: $src_file"
      fi
    done
  fi
done
