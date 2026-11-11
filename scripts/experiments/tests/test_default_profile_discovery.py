#!/usr/bin/env python3
"""Check default bank discovery, explicit opt-out and incompatible modes."""
from pathlib import Path
import os,subprocess,tempfile,json
ROOT=Path(__file__).resolve().parents[3]
source=r'''
#include "epmapping/defaults.hpp"
#include <cassert>
int main(int argc,char** argv){
  auto const lib=std::filesystem::path{argv[1]};
  unsetenv("MAP_EGG_PROFILE_READONLY_DIR");
  setenv("MAP_EGG_PROFILE","1",1);setenv("MAP_EGG_PROFILE_TT_CACHE","0",1);
  epmapping::apply_verified_candidate_library(lib);
  bool installed=std::filesystem::is_regular_file(std::filesystem::path{argv[2]}/"library_manifest.json");
  assert(bool(getenv("MAP_EGG_PROFILE_READONLY_DIR"))==installed);
  if(installed)assert(std::filesystem::weakly_canonical(getenv("MAP_EGG_PROFILE_READONLY_DIR"))==std::filesystem::weakly_canonical(argv[2]));
  setenv("MAP_EGG_PROFILE_READONLY_DIR","",1);epmapping::apply_verified_candidate_library(lib);
  assert(std::string(getenv("MAP_EGG_PROFILE_READONLY_DIR")).empty());
  setenv("MAP_EGG_PROFILE_READONLY_DIR","explicit-location",1);epmapping::apply_verified_candidate_library(lib);
  assert(std::string(getenv("MAP_EGG_PROFILE_READONLY_DIR"))=="explicit-location");
  unsetenv("MAP_EGG_PROFILE_READONLY_DIR");epmapping::apply_verified_candidate_library("unknown.genlib");
  assert(!getenv("MAP_EGG_PROFILE_READONLY_DIR"));
  setenv("MAP_EGG_PROFILE","0",1);epmapping::apply_verified_candidate_library(lib);
  assert(!getenv("MAP_EGG_PROFILE_READONLY_DIR"));
  setenv("MAP_EGG_PROFILE","1",1);setenv("MAP_EGG_PROFILE_TT_CACHE","1",1);
  epmapping::apply_verified_candidate_library(lib);assert(!getenv("MAP_EGG_PROFILE_READONLY_DIR"));
}
'''
with tempfile.TemporaryDirectory(prefix='profile-discovery-') as tmp:
 p=Path(tmp);(p/'test.cpp').write_text(source)
 subprocess.run(['g++','-std=c++17','-I',str(ROOT),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
 subprocess.run([str(p/'test'),str(ROOT/'epmapping/libraries/asap7.genlib'),str(ROOT/'data/epmapping/prebuilt/full')],check=True)
print(json.dumps(dict(passed=True,automatic_and_explicit_modes=True,unknown_library_and_incompatible_modes=True)))
