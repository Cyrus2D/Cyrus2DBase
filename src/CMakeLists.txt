
add_subdirectory(player)
add_subdirectory(coach)
add_subdirectory(trainer)

# generate shell scripts and copy them to the binary direcotry as an executable script
configure_file(
  ${PROJECT_SOURCE_DIR}/src/start.sh.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/start.sh
  @ONLY
  )
file(
  COPY ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/start.sh
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

configure_file(
  ${PROJECT_SOURCE_DIR}/src/keepaway.sh.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/keepaway.sh
  @ONLY
  )
file(
  COPY ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/keepaway.sh
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

configure_file(
  ${PROJECT_SOURCE_DIR}/src/train.sh.in
  ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/train.sh
  @ONLY
  )
file(
  COPY ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/train.sh
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )

# copy other files to the binary direcotry
file(COPY
  formations-dt formations-keeper formations-taker player.conf coach.conf start-debug.sh start-offline.sh unmark_dnn_weights.txt
  DESTINATION ${PROJECT_BINARY_DIR}/bin
  )
