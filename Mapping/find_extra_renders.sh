#!/usr/bin/env bash
# find_extra_renders.sh
# Finds render folders that have no matching .obj in the curated source.

OBJECTS_ROOT=~/data/objects
RENDERS_ROOT=~/data/renders

check_type() {
    local sym_type=$1        # axis_sym or plane_sym
    local obj_dir=$2         # curated_axis_sym_obj or curated_plane_sym_obj

    local obj_path="$OBJECTS_ROOT/$obj_dir"
    local render_path="$RENDERS_ROOT/$sym_type"

    # Build set of valid IDs from .obj filenames (strip extension)
    local tmp_valid
    tmp_valid=$(mktemp)
    ls "$obj_path"/*.obj 2>/dev/null | xargs -I{} basename {} .obj | sort > "$tmp_valid"

    # Build set of render folder names (directories only, exclude files)
    local tmp_renders
    tmp_renders=$(mktemp)
    find "$render_path" -maxdepth 1 -mindepth 1 -type d | xargs -I{} basename {} | sort > "$tmp_renders"

    local n_valid
    local n_renders
    n_valid=$(wc -l < "$tmp_valid")
    n_renders=$(wc -l < "$tmp_renders")

    echo "======================================"
    echo "  $sym_type"
    echo "======================================"
    echo "  .obj fuente : $n_valid"
    echo "  Render dirs : $n_renders"
    echo "  Extra       : $((n_renders - n_valid))"
    echo ""
    echo "  Carpetas SIN .obj correspondiente:"
    comm -23 "$tmp_renders" "$tmp_valid" | while read -r id; do
        render_dir="$render_path/$id"
        n_images=$(find "$render_dir" -name "*.png" 2>/dev/null | wc -l)
        echo "    $id  (${n_images} PNGs)"
    done
    echo ""

    rm -f "$tmp_valid" "$tmp_renders"
}

check_type "axis_sym"  "curated_axis_sym_obj"
check_type "plane_sym" "curated_plane_sym_obj"
