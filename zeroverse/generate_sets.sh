#!/bin/bash

OUTPUT_DIR="/home/vaclav_knapp/zeroverse/berkeley_test_2"
#TEXTURE_FOLDER="/workspace/UC_Berkeley/Voronoi/textures"
TRANSLATION_CONTROL=0.5
NUM_SHAPES=1


generate_random_shape_counts() {
  local n_objects=$1
  local CYLINDER_COUNT
  local CUBE_COUNT
  local ELLIPSOID_COUNT

  while true; do

    CYLINDER_COUNT=$(( RANDOM % (n_objects + 1) ))
    CUBE_COUNT=$(( RANDOM % (n_objects + 1) ))
    ELLIPSOID_COUNT=$(( n_objects - CYLINDER_COUNT - CUBE_COUNT ))


    if [ "$ELLIPSOID_COUNT" -ge 0 ]; then
      break
    fi
  done

  SHAPE_COUNTS="cylinder:$CYLINDER_COUNT,cube:$CUBE_COUNT,ellipsoid:$ELLIPSOID_COUNT"

  echo "$SHAPE_COUNTS"
}


for n_objects in {2..4}
do
  echo "Processing n_objects = $n_objects"


  SIZE_SEED=$(( RANDOM ))


  for j in {1..15}
  do

    SHAPE_COUNTS=$(generate_random_shape_counts $n_objects)


    DIR_NAME="example_${j}_$(echo $SHAPE_COUNTS | sed 's/,/_/g')"


    CONFIG_OUTPUT_DIR="${OUTPUT_DIR}/${n_objects}/${DIR_NAME}"
    mkdir -p "$CONFIG_OUTPUT_DIR"

    echo "  Shape counts: $SHAPE_COUNTS"


    for k in {1..10}
    do

      SEED=$(( RANDOM ))

      python /home/vaclav_knapp/zeroverse/create_shape_blend.py --num_shapes $NUM_SHAPES \
                                                                --output_dir "$CONFIG_OUTPUT_DIR" \
                                                                --shape_counts "$SHAPE_COUNTS" \
                                                                --seed $SEED \
                                                                --size_seed $SIZE_SEED
                                                                #--translation_control $TRANSLATION_CONTROL \

      # Define the object name format (object_{k}.blend)
      OBJECT_NAME="object_${k}.blend"
      OUTPUT_FILE="${CONFIG_OUTPUT_DIR}/${OBJECT_NAME}"

      mv "${CONFIG_OUTPUT_DIR}/object_000.blend" "$OUTPUT_FILE"
    done
  done
done

echo "All shape configurations, examples, and objects have been generated with random seeds!"
