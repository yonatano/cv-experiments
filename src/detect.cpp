#include "detect.h"

using namespace arma;

bool isKeypoint(Row<int> X) {
    if (X[0] == 0) {
      return 0;
     }
     if (X[0] == 1) {
      if (X[6] == 1) {
       if (X[5] == 0) {
        if (X[7] == 0) {
         if (X[8] == 0) {
          return 0;
         }
         if (X[8] == 1) {
          if (X[10] == 1) {
           if (X[1] == 0) {
            return 0;
           }
           if (X[1] == 1) {
            if (X[9] == 1) {
             if (X[11] == 0) {
              return 0;
             }
             if (X[11] == 1) {
              if (X[13] == 1) {
               if (X[12] == 0) {
                return 0;
               }
               if (X[12] == 1) {
                if (X[2] == 0) {
                 return 0;
                }
                if (X[2] == 1) {
                 if (X[3] == 0) {
                  return 0;
                 }
                 if (X[3] == 1) {
                  return 1;
                 }
                 if (X[3] == 2) {
                  return 0;
                 }
                }
                if (X[2] == 2) {
                 return 0;
                }
               }
               if (X[12] == 2) {
                return 0;
               }
              }
             }
             if (X[11] == 2) {
              return 0;
             }
            }
           }
           if (X[1] == 2) {
            return 0;
           }
          }
         }
         if (X[8] == 2) {
          if (X[10] == 2) {
           if (X[11] == 0) {
            return 0;
           }
           if (X[11] == 1) {
            return 0;
           }
           if (X[11] == 2) {
            if (X[13] == 2) {
             if (X[12] == 0) {
              return 0;
             }
             if (X[12] == 1) {
              return 0;
             }
             if (X[12] == 2) {
              if (X[14] == 2) {
               if (X[3] == 0) {
                return 0;
               }
               if (X[3] == 1) {
                return 0;
               }
               if (X[3] == 2) {
                if (X[15] == 2) {
                 if (X[2] == 0) {
                  return 0;
                 }
                 if (X[2] == 1) {
                  return 0;
                 }
                 if (X[2] == 2) {
                  if (X[1] == 0) {
                   return 0;
                  }
                  if (X[1] == 1) {
                   return 0;
                  }
                  if (X[1] == 2) {
                   return 1;
                  }
                 }
                }
               }
              }
             }
            }
           }
          }
         }
        }
       }
       if (X[5] == 1) {
        if (X[7] == 1) {
         if (X[11] == 0) {
          if (X[13] == 0) {
           if (X[2] == 0) {
            return 0;
           }
           if (X[2] == 1) {
            if (X[4] == 1) {
             if (X[3] == 0) {
              return 0;
             }
             if (X[3] == 1) {
              if (X[8] == 0) {
               return 0;
              }
              if (X[8] == 1) {
               if (X[1] == 0) {
                return 0;
               }
               if (X[1] == 1) {
                return 1;
               }
               if (X[1] == 2) {
                return 0;
               }
              }
              if (X[8] == 2) {
               return 0;
              }
             }
             if (X[3] == 2) {
              return 0;
             }
            }
           }
           if (X[2] == 2) {
            return 0;
           }
          }
         }
         if (X[11] == 1) {
          if (X[13] == 1) {
           if (X[12] == 0) {
            if (X[14] == 0) {
             if (X[1] == 0) {
              return 0;
             }
             if (X[1] == 1) {
              if (X[9] == 1) {
               if (X[3] == 0) {
                return 0;
               }
               if (X[3] == 1) {
                if (X[8] == 0) {
                 return 0;
                }
                if (X[8] == 1) {
                 if (X[2] == 0) {
                  return 0;
                 }
                 if (X[2] == 1) {
                  return 1;
                 }
                 if (X[2] == 2) {
                  return 0;
                 }
                }
                if (X[8] == 2) {
                 return 0;
                }
               }
               if (X[3] == 2) {
                return 0;
               }
              }
             }
             if (X[1] == 2) {
              return 0;
             }
            }
           }
           if (X[12] == 1) {
            if (X[14] == 1) {
             if (X[1] == 0) {
              if (X[3] == 0) {
               return 0;
              }
              if (X[3] == 1) {
               if (X[2] == 0) {
                return 0;
               }
               if (X[2] == 1) {
                return 1;
               }
               if (X[2] == 2) {
                return 0;
               }
              }
              if (X[3] == 2) {
               return 0;
              }
             }
             if (X[1] == 1) {
              if (X[9] == 1) {
               if (X[3] == 0) {
                if (X[8] == 0) {
                 return 0;
                }
                if (X[8] == 1) {
                 return 1;
                }
                if (X[8] == 2) {
                 return 0;
                }
               }
               if (X[3] == 1) {
                if (X[15] == 1) {
                 if (X[8] == 0) {
                  if (X[2] == 0) {
                   return 0;
                  }
                  if (X[2] == 1) {
                   return 1;
                  }
                  if (X[2] == 2) {
                   return 0;
                  }
                 }
                 if (X[8] == 1) {
                  return 1;
                 }
                 if (X[8] == 2) {
                  if (X[2] == 0) {
                   return 0;
                  }
                  if (X[2] == 1) {
                   return 1;
                  }
                  if (X[2] == 2) {
                   return 0;
                  }
                 }
                }
               }
               if (X[3] == 2) {
                if (X[8] == 0) {
                 return 0;
                }
                if (X[8] == 1) {
                 return 1;
                }
                if (X[8] == 2) {
                 return 0;
                }
               }
              }
             }
             if (X[1] == 2) {
              if (X[3] == 0) {
               return 0;
              }
              if (X[3] == 1) {
               if (X[2] == 0) {
                return 0;
               }
               if (X[2] == 1) {
                return 1;
               }
               if (X[2] == 2) {
                return 0;
               }
              }
              if (X[3] == 2) {
               return 0;
              }
             }
            }
           }
           if (X[12] == 2) {
            if (X[14] == 2) {
             if (X[1] == 0) {
              return 0;
             }
             if (X[1] == 1) {
              if (X[9] == 1) {
               if (X[3] == 0) {
                return 0;
               }
               if (X[3] == 1) {
                if (X[8] == 0) {
                 return 0;
                }
                if (X[8] == 1) {
                 if (X[2] == 0) {
                  return 0;
                 }
                 if (X[2] == 1) {
                  return 1;
                 }
                 if (X[2] == 2) {
                  return 0;
                 }
                }
                if (X[8] == 2) {
                 return 0;
                }
               }
               if (X[3] == 2) {
                return 0;
               }
              }
             }
             if (X[1] == 2) {
              return 0;
             }
            }
           }
          }
         }
         if (X[11] == 2) {
          if (X[13] == 2) {
           if (X[2] == 0) {
            return 0;
           }
           if (X[2] == 1) {
            if (X[4] == 1) {
             if (X[3] == 0) {
              return 0;
             }
             if (X[3] == 1) {
              if (X[15] == 1) {
               if (X[8] == 0) {
                return 0;
               }
               if (X[8] == 1) {
                if (X[1] == 0) {
                 return 0;
                }
                if (X[1] == 1) {
                 return 1;
                }
                if (X[1] == 2) {
                 return 0;
                }
               }
               if (X[8] == 2) {
                return 0;
               }
              }
             }
             if (X[3] == 2) {
              return 0;
             }
            }
           }
           if (X[2] == 2) {
            if (X[4] == 2) {
             if (X[3] == 0) {
              return 0;
             }
             if (X[3] == 1) {
              return 0;
             }
             if (X[3] == 2) {
              if (X[15] == 2) {
               if (X[12] == 0) {
                return 0;
               }
               if (X[12] == 1) {
                return 0;
               }
               if (X[12] == 2) {
                if (X[14] == 2) {
                 if (X[8] == 0) {
                  return 0;
                 }
                 if (X[8] == 1) {
                  return 0;
                 }
                 if (X[8] == 2) {
                  if (X[1] == 0) {
                   return 0;
                  }
                  if (X[1] == 1) {
                   return 0;
                  }
                  if (X[1] == 2) {
                   return 1;
                  }
                 }
                }
               }
              }
             }
            }
           }
          }
         }
        }
       }
       if (X[5] == 2) {
        if (X[7] == 2) {
         if (X[8] == 0) {
          return 0;
         }
         if (X[8] == 1) {
          if (X[10] == 1) {
           if (X[1] == 0) {
            return 0;
           }
           if (X[1] == 1) {
            if (X[9] == 1) {
             if (X[11] == 0) {
              return 0;
             }
             if (X[11] == 1) {
              if (X[13] == 1) {
               if (X[12] == 0) {
                return 0;
               }
               if (X[12] == 1) {
                if (X[2] == 0) {
                 return 0;
                }
                if (X[2] == 1) {
                 if (X[3] == 0) {
                  return 0;
                 }
                 if (X[3] == 1) {
                  return 1;
                 }
                 if (X[3] == 2) {
                  return 0;
                 }
                }
                if (X[2] == 2) {
                 return 0;
                }
               }
               if (X[12] == 2) {
                return 0;
               }
              }
             }
             if (X[11] == 2) {
              return 0;
             }
            }
           }
           if (X[1] == 2) {
            return 0;
           }
          }
         }
         if (X[8] == 2) {
          if (X[10] == 2) {
           if (X[11] == 0) {
            return 0;
           }
           if (X[11] == 1) {
            return 0;
           }
           if (X[11] == 2) {
            if (X[13] == 2) {
             if (X[3] == 0) {
              return 0;
             }
             if (X[3] == 1) {
              return 0;
             }
             if (X[3] == 2) {
              if (X[15] == 2) {
               if (X[12] == 0) {
                return 0;
               }
               if (X[12] == 1) {
                return 0;
               }
               if (X[12] == 2) {
                if (X[1] == 0) {
                 return 0;
                }
                if (X[1] == 1) {
                 return 0;
                }
                if (X[1] == 2) {
                 if (X[2] == 0) {
                  return 0;
                 }
                 if (X[2] == 1) {
                  return 0;
                 }
                 if (X[2] == 2) {
                  return 1;
                 }
                }
               }
              }
             }
            }
           }
          }
         }
        }
       }
      }
     }
     if (X[0] == 2) {
      if (X[6] == 2) {
       if (X[5] == 0) {
        return 0;
       }
       if (X[5] == 1) {
        return 0;
       }
       if (X[5] == 2) {
        if (X[7] == 2) {
         if (X[1] == 0) {
          if (X[9] == 0) {
           if (X[2] == 0) {
            return 0;
           }
           if (X[2] == 1) {
            return 0;
           }
           if (X[2] == 2) {
            if (X[4] == 2) {
             if (X[3] == 0) {
              return 0;
             }
             if (X[3] == 1) {
              return 0;
             }
             if (X[3] == 2) {
              if (X[15] == 2) {
               if (X[12] == 0) {
                return 0;
               }
               if (X[12] == 1) {
                return 0;
               }
               if (X[12] == 2) {
                if (X[14] == 2) {
                 if (X[11] == 0) {
                  return 0;
                 }
                 if (X[11] == 1) {
                  return 0;
                 }
                 if (X[11] == 2) {
                  if (X[8] == 0) {
                   return 0;
                  }
                  if (X[8] == 1) {
                   return 1;
                  }
                  if (X[8] == 2) {
                   return 0;
                  }
                 }
                }
               }
              }
             }
            }
           }
          }
         }
         if (X[1] == 1) {
          if (X[9] == 1) {
           if (X[8] == 0) {
            return 0;
           }
           if (X[8] == 1) {
            if (X[10] == 1) {
             if (X[2] == 0) {
              return 0;
             }
             if (X[2] == 1) {
              return 0;
             }
             if (X[2] == 2) {
              if (X[4] == 2) {
               if (X[3] == 0) {
                return 0;
               }
               if (X[3] == 1) {
                return 0;
               }
               if (X[3] == 2) {
                if (X[15] == 2) {
                 if (X[11] == 0) {
                  return 0;
                 }
                 if (X[11] == 1) {
                  return 0;
                 }
                 if (X[11] == 2) {
                  if (X[12] == 0) {
                   return 0;
                  }
                  if (X[12] == 1) {
                   return 0;
                  }
                  if (X[12] == 2) {
                   return 1;
                  }
                 }
                }
               }
              }
             }
            }
           }
           if (X[8] == 2) {
            return 0;
           }
          }
         }
         if (X[1] == 2) {
          if (X[9] == 2) {
           if (X[11] == 0) {
            if (X[13] == 0) {
             if (X[2] == 0) {
              return 0;
             }
             if (X[2] == 1) {
              return 0;
             }
             if (X[2] == 2) {
              if (X[4] == 2) {
               if (X[8] == 0) {
                return 0;
               }
               if (X[8] == 1) {
                return 0;
               }
               if (X[8] == 2) {
                if (X[10] == 2) {
                 if (X[3] == 0) {
                  return 0;
                 }
                 if (X[3] == 1) {
                  return 0;
                 }
                 if (X[3] == 2) {
                  if (X[12] == 0) {
                   return 0;
                  }
                  if (X[12] == 1) {
                   return 1;
                  }
                  if (X[12] == 2) {
                   return 0;
                  }
                 }
                }
               }
              }
             }
            }
           }
           if (X[11] == 1) {
            if (X[13] == 1) {
             if (X[12] == 0) {
              return 0;
             }
             if (X[12] == 1) {
              if (X[14] == 1) {
               if (X[2] == 0) {
                return 0;
               }
               if (X[2] == 1) {
                return 0;
               }
               if (X[2] == 2) {
                if (X[4] == 2) {
                 if (X[3] == 0) {
                  return 0;
                 }
                 if (X[3] == 1) {
                  return 0;
                 }
                 if (X[3] == 2) {
                  if (X[8] == 0) {
                   return 0;
                  }
                  if (X[8] == 1) {
                   return 0;
                  }
                  if (X[8] == 2) {
                   return 1;
                  }
                 }
                }
               }
              }
             }
             if (X[12] == 2) {
              return 0;
             }
            }
           }
           if (X[11] == 2) {
            if (X[13] == 2) {
             if (X[3] == 0) {
              if (X[15] == 0) {
               if (X[8] == 0) {
                return 0;
               }
               if (X[8] == 1) {
                return 0;
               }
               if (X[8] == 2) {
                if (X[10] == 2) {
                 if (X[12] == 0) {
                  return 0;
                 }
                 if (X[12] == 1) {
                  return 0;
                 }
                 if (X[12] == 2) {
                  if (X[2] == 0) {
                   return 0;
                  }
                  if (X[2] == 1) {
                   return 1;
                  }
                  if (X[2] == 2) {
                   return 0;
                  }
                 }
                }
               }
              }
             }
             if (X[3] == 1) {
              if (X[15] == 1) {
               if (X[8] == 0) {
                return 0;
               }
               if (X[8] == 1) {
                return 0;
               }
               if (X[8] == 2) {
                if (X[10] == 2) {
                 if (X[12] == 0) {
                  return 0;
                 }
                 if (X[12] == 1) {
                  return 0;
                 }
                 if (X[12] == 2) {
                  if (X[2] == 0) {
                   return 0;
                  }
                  if (X[2] == 1) {
                   return 1;
                  }
                  if (X[2] == 2) {
                   return 0;
                  }
                 }
                }
               }
              }
             }
             if (X[3] == 2) {
              if (X[15] == 2) {
               if (X[2] == 0) {
                return 0;
               }
               if (X[2] == 1) {
                if (X[8] == 0) {
                 return 0;
                }
                if (X[8] == 1) {
                 return 0;
                }
                if (X[8] == 2) {
                 if (X[12] == 0) {
                  return 0;
                 }
                 if (X[12] == 1) {
                  return 0;
                 }
                 if (X[12] == 2) {
                  return 1;
                 }
                }
               }
               if (X[2] == 2) {
                if (X[4] == 2) {
                 if (X[12] == 0) {
                  return 0;
                 }
                 if (X[12] == 1) {
                  if (X[8] == 0) {
                   return 0;
                  }
                  if (X[8] == 1) {
                   return 0;
                  }
                  if (X[8] == 2) {
                   return 1;
                  }
                 }
                 if (X[12] == 2) {
                  if (X[8] == 0) {
                   return 0;
                  }
                  if (X[8] == 1) {
                   return 1;
                  }
                  if (X[8] == 2) {
                   return 0;
                  }
                 }
                }
               }
              }
             }
            }
           }
          }
         }
        }
       }
      }
     }
     return 0;
}