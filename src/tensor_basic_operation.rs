use std::fmt::Display;

use typenum::Len;

use crate::{index::{TensorIndex, TensorIndexUncheck}, 
            ERIFull, ERIFold4, MatrixFull, MatrixFullSliceMut, 
            MatrixFullSlice, MatrixUpperSliceMut, MatrixUpper, 
            MatrixUpperSlice};

/// Trait definitions for tensor basic operations, mainly including
///    getting a (mutable) number, or a (mutable) slice from a defined tensor
pub trait TensorOpt<T> where Self: TensorIndex {
    fn get1d(&self, position:usize) -> Option<&T> {None}
    fn get2d(&self, position:[usize;2]) -> Option<&T> {None}
    fn get3d(&self, position:[usize;3]) -> Option<&T> {None}
    fn get4d(&self, position:[usize;4]) -> Option<&T> {None}
    fn get(&self, position:&[usize]) -> Option<&T> {None}
}

pub trait TensorOptUncheck<T> where Self: TensorIndexUncheck {
    // For MatrixUpper
    fn get2d_uncheck(&self, position:[usize;2]) -> Option<&T> {None}
    // For ERIFold4
    fn get4d_uncheck(&self, position:[usize;4]) -> Option<&T> {None}
}

pub trait TensorOptMut<'a, T> where Self: TensorIndex {
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {None}
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {None}
    fn get3d_mut(&mut self, position:[usize;3]) -> Option<&mut T> {None}
    fn get4d_mut(&mut self, position:[usize;4]) -> Option<&mut T> {None}
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {None}

    fn set1d(&mut self, position:usize, new_data: T) {}
    fn set2d(&mut self, position:[usize;2], new_data: T) {}
    fn set3d(&mut self, position:[usize;3], new_data: T) {}
    fn set4d(&mut self, position:[usize;4], new_data: T) {}
    fn set(&mut self, position:&[usize], new_data: T) {}
}

pub trait TensorOptMutUncheck<'a, T> where Self: TensorIndexUncheck {
    // For MatrixUpperMut
    fn get2d_mut_uncheck(&mut self, position:[usize;2]) -> Option<&mut T> {None}
    // For ERIFold4
    fn get4d_mut_uncheck(&mut self, position:[usize;4]) -> Option<&mut T> {None}
    // For MatrixUpperMut
    fn set2d_uncheck(&mut self, position:[usize;2], new_data: T) {}
    // For ERIFold4
    fn set4d_uncheck(&mut self, position:[usize;4], new_data: T) {}
}

pub trait TensorSlice<T> where Self: TensorIndex+TensorOpt<T> {
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {None}
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {None}
    fn get3d_slice(&self, position:[usize;3], length: usize) -> Option<&[T]> {None}
    fn get4d_slice(&self, position:[usize;4], length: usize) -> Option<&[T]> {None}
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {None}
}

pub trait TensorSliceUncheck<T> where Self: TensorIndexUncheck+TensorOptUncheck<T> {
    // For MatrixUpper
    fn get2d_slice_uncheck(&self, position:[usize;2],length: usize) -> Option<&[T]> {None}
    // For ERIFold4
    fn get4d_slice_uncheck(&self, position:[usize;4],length: usize) -> Option<&[T]> {None}
}
pub trait TensorSliceMut<'a, T> where Self: TensorIndex+TensorOptMut<'a,T> {
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {None}
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {None}
    fn get3d_slice_mut(&mut self, position:[usize;3], length: usize) -> Option<&mut [T]> {None}
    fn get4d_slice_mut(&mut self, position:[usize;4], length: usize) -> Option<&mut [T]> {None}
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {None}
}

/// Define for upper-formated tensors specifically, like 
///  ERIFold4 with the indice of [i,j,k,l], and
///  MatrixUpper with the indice of [i,j].
/// According to the upper-formated tensor definition, in principle, i=<j and k<=l.
/// WARNING: For efficiency, "uncheck" here means that we don't check the index order during the tensor operations.
/// WARNING: As a result, a wrong index order could lead to a wrong operation.
pub trait TensorSliceMutUncheck<'a, T> where Self: TensorIndexUncheck+TensorOptMutUncheck<'a,T> {
    // For MatrixUpperMut
    fn get2d_slice_mut_uncheck(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {None}
    // For ERIFold4
    fn get4d_slice_mut_uncheck(&mut self, position:[usize;4], length: usize) -> Option<&mut [T]> {None}
}


/// Implementation of the traits for specific tensor structures
impl<T: Clone+Display> TensorOpt<T> for ERIFull<T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get4d(&self, position:[usize;4]) -> Option<&T> {
        self.data.get(self.index4d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1],position[2],position[3]];
        self.data.get(self.index4d(tp).unwrap())
    }
}
impl<'a, T: Clone+Display> TensorOptMut<'a, T> for ERIFull<T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get4d_mut(&mut self, position:[usize;4]) -> Option<&mut T> {
        let tp = self.index4d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index4d([position[0],position[1],position[2],position[3]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set4d(&mut self, position:[usize;4], new_data: T) {
        let tp = self.index4d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        //let tp = self.index4d([position[0],position[1],position[2],position[3]]);
        self.set4d([position[0],position[1],position[2],position[3]], new_data);
        
    }
}

impl<T: Clone+Display> TensorSlice<T> for ERIFull<T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get4d_slice(&self, position:[usize;4], length: usize) -> Option<&[T]> {
        let tp = self.index4d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get4d_slice([position[0],position[1],position[2],position[3]], length)
    }
}
impl<'a, T: Clone+Display> TensorSliceMut<'a,T> for ERIFull<T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get4d_slice_mut(&mut self, position:[usize;4], length: usize) -> Option<&mut [T]> {
        let tp = self.index4d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get4d_slice_mut([position[0],position[1],position[2],position[3]], length)
    }
}

/// For ERIFold4
impl<T: Clone+Display> TensorOpt<T> for ERIFold4<T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get2d(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d(position).unwrap())
    }
    #[inline]
    fn get4d(&self, position:[usize;4]) -> Option<&T> {
        self.data.get(self.index4d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1],position[2],position[3]];
        self.data.get(self.index4d(tp).unwrap())
    }

    fn get3d(&self, position:[usize;3]) -> Option<&T> {None}
}
impl<T: Clone+Display> TensorOptUncheck<T> for ERIFold4<T> {
    #[inline]
    fn get4d_uncheck(&self, position:[usize;4]) -> Option<&T> {
        self.data.get(self.index4d_uncheck(position).unwrap())
    }
}
impl<'a, T: Clone+Display> TensorOptMut<'a, T> for ERIFold4<T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get4d_mut(&mut self, position:[usize;4]) -> Option<&mut T> {
        let tp = self.index4d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index4d([position[0],position[1],position[2],position[3]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set2d(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set4d(&mut self, position:[usize;4], new_data: T) {
        let tp = self.index4d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        self.set4d([position[0],position[1],position[2],position[3]], new_data);
        
    }
}
impl<'a, T: Clone+Display> TensorOptMutUncheck<'a, T> for ERIFold4<T> {
    #[inline]
    fn set4d_uncheck(&mut self, position:[usize;4], new_data: T) {
        let tp = self.index4d_uncheck(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn get4d_mut_uncheck(&mut self, position:[usize;4]) -> Option<&mut T> {
        let tp = self.index4d_uncheck(position).unwrap();
        self.data.get_mut(tp)
    }
}

impl<T: Clone+Display> TensorSlice<T> for ERIFold4<T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get4d_slice(&self, position:[usize;4], length: usize) -> Option<&[T]> {
        let tp = self.index4d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get4d_slice([position[0],position[1],position[2],position[3]], length)
    }
}

impl<T: Clone+Display> TensorSliceUncheck<T> for ERIFold4<T> {
    #[inline]
    fn get4d_slice_uncheck(&self, position:[usize;4], length: usize) -> Option<&[T]> {
        let tp = self.index4d_uncheck(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
}

impl<'a, T: Clone+Display> TensorSliceMut<'a,T> for ERIFold4<T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get4d_slice_mut(&mut self, position:[usize;4], length: usize) -> Option<&mut [T]> {
        let tp = self.index4d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get4d_slice_mut([position[0],position[1],position[2],position[3]], length)
    }
}

impl<'a, T: Clone+Display> TensorSliceMutUncheck<'a,T> for ERIFold4<T> {
    #[inline]
    fn get4d_slice_mut_uncheck(&mut self, position:[usize;4], length: usize) -> Option<&mut [T]> {
        let tp = self.index4d_uncheck(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
}

// Trait implementations for MatrixFull
impl<'a,T: Clone+Display+Send+Sync> TensorOpt<T> for MatrixFull<T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get2d(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1]];
        self.data.get(self.index2d(tp).unwrap())
    }
}
impl<'a, T: Clone+Display+Send+Sync> TensorSlice<T> for MatrixFull<T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get2d_slice([position[0],position[1]], length)
    }
}

impl<'a, T: Clone+Display+Send+Sync> TensorOptMut<'a, T> for MatrixFull<T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index2d([position[0],position[1]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set2d(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        //let tp = self.index4d([position[0],position[1],position[2],position[3]]);
        self.set2d([position[0],position[1]], new_data);
        
    }
}
impl<'a, T: Clone+Display+Send+Sync> TensorSliceMut<'a, T> for MatrixFull<T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get2d_slice_mut([position[0],position[1]], length)
    }
}

// Trait implementations for MatrixFullSliceMut
impl<'a, T: Clone+Display> TensorOptMut<'a, T> for MatrixFullSliceMut<'a,T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index2d([position[0],position[1]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set2d(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        //let tp = self.index4d([position[0],position[1],position[2],position[3]]);
        self.set2d([position[0],position[1]], new_data);
        
    }
}
impl<'a, T: Clone+Display> TensorSliceMut<'a, T> for MatrixFullSliceMut<'a, T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get2d_slice_mut([position[0],position[1]], length)
    }
}

//Trait implementations for MatrixFullSlice
impl<'a,T: Clone+Display> TensorOpt<T> for MatrixFullSlice<'a,T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get2d(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1]];
        self.data.get(self.index2d(tp).unwrap())
    }
}
impl<'a, T: Clone+Display> TensorSlice<T> for MatrixFullSlice<'a, T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get2d_slice([position[0],position[1]], length)
    }
}

//Trait implementations for MatrixUpperSliceMut
impl<'a, T: Clone+Display> TensorOptMut<'a, T> for MatrixUpperSliceMut<'a,T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index2d([position[0],position[1]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set2d(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        //let tp = self.index4d([position[0],position[1],position[2],position[3]]);
        self.set2d([position[0],position[1]], new_data);
        
    }
}
impl<'a, T: Clone+Display> TensorOptMutUncheck<'a, T> for MatrixUpperSliceMut<'a,T> {
    #[inline]
    fn get2d_mut_uncheck(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d_uncheck(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set2d_uncheck(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d_uncheck(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
}
impl<'a, T: Clone+Display> TensorSliceMut<'a, T> for MatrixUpperSliceMut<'a, T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get2d_slice_mut([position[0],position[1]], length)
    }
}
impl<'a, T: Clone+Display> TensorSliceMutUncheck<'a, T> for MatrixUpperSliceMut<'a, T> {
    #[inline]
    fn get2d_slice_mut_uncheck(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d_uncheck(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
}

//Trait implementations for MatrixUpperSlice
impl<'a,T: Clone+Display> TensorOpt<T> for MatrixUpperSlice<'a,T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get2d(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1]];
        self.data.get(self.index2d(tp).unwrap())
    }
}
impl<'a,T: Clone+Display> TensorOptUncheck<T> for MatrixUpperSlice<'a,T> {
    #[inline]
    fn get2d_uncheck(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d_uncheck(position).unwrap())
    }
}
impl<'a, T: Clone+Display> TensorSlice<T> for MatrixUpperSlice<'a, T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get2d_slice([position[0],position[1]], length)
    }
}
impl<'a, T: Clone+Display> TensorSliceUncheck<T> for MatrixUpperSlice<'a, T> {
    #[inline]
    fn get2d_slice_uncheck(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d_uncheck(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
}

//Trait implementations for MatrixUpper
impl<'a, T: Clone+Display> TensorOptMut<'a, T> for MatrixUpper<T> {
    #[inline]
    fn get1d_mut(&mut self, position:usize) -> Option<&mut T> {
        self.data.get_mut(position)
    }
    #[inline]
    fn get2d_mut(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn get_mut(&mut self, position:&[usize]) -> Option<&mut T> {
        let tp = self.index2d([position[0],position[1]]).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set1d(&mut self, position:usize, new_data: T) {
        if let Some(tmp_value) = self.data.get_mut(position) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set2d(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
    #[inline]
    fn set(&mut self, position:&[usize], new_data: T) {
        //let tp = self.index4d([position[0],position[1],position[2],position[3]]);
        self.set2d([position[0],position[1]], new_data);
        
    }
}
impl<'a, T: Clone+Display> TensorOptMutUncheck<'a, T> for MatrixUpper<T> {
    #[inline]
    fn get2d_mut_uncheck(&mut self, position:[usize;2]) -> Option<&mut T> {
        let tp = self.index2d_uncheck(position).unwrap();
        self.data.get_mut(tp)
    }
    #[inline]
    fn set2d_uncheck(&mut self, position:[usize;2], new_data: T) {
        let tp = self.index2d_uncheck(position).unwrap();
        if let Some(tmp_value) = self.data.get_mut(tp) {
            *tmp_value = new_data
        } else {
          panic!("Error in setting the tensor element located at the position of {:?}", position);
        };
    }
}
impl<'a, T: Clone+Display> TensorSliceMut<'a, T> for MatrixUpper<T> {
    #[inline]
    fn get1d_slice_mut(&mut self, position:usize, length: usize) -> Option<&mut [T]> {
        Some(&mut self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice_mut(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice_mut(&mut self, position:&[usize], length: usize) -> Option<&mut [T]> {
        self.get2d_slice_mut([position[0],position[1]], length)
    }
}
impl<'a, T: Clone+Display> TensorSliceMutUncheck<'a, T> for MatrixUpper<T> {
    #[inline]
    fn get2d_slice_mut_uncheck(&mut self, position:[usize;2], length: usize) -> Option<&mut [T]> {
        let tp = self.index2d_uncheck(position).unwrap();
        Some(&mut self.data[tp..tp+length])
    }
}
impl<T: Clone+Display> TensorOpt<T> for MatrixUpper<T> {
    #[inline]
    fn get1d(&self, position:usize) -> Option<&T> {
        self.data.get(position)
    }
    #[inline]
    fn get2d(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d(position).unwrap())
    }
    #[inline]
    fn get(&self, position:&[usize]) -> Option<&T> {
        let tp = [position[0],position[1]];
        self.data.get(self.index2d(tp).unwrap())
    }
}
impl<T: Clone+Display> TensorOptUncheck<T> for MatrixUpper<T> {
    #[inline]
    fn get2d_uncheck(&self, position:[usize;2]) -> Option<&T> {
        self.data.get(self.index2d_uncheck(position).unwrap())
    }
}
impl<T: Clone+Display> TensorSlice<T> for MatrixUpper<T> {
    #[inline]
    fn get1d_slice(&self, position:usize, length: usize) -> Option<&[T]> {
        Some(&self.data[position..position+length])
    }
    #[inline]
    fn get2d_slice(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
    #[inline]
    fn get_slice(&self, position:&[usize], length: usize) -> Option<&[T]> {
        self.get2d_slice([position[0],position[1]], length)
    }
}
impl<T: Clone+Display> TensorSliceUncheck<T> for MatrixUpper<T> {
    #[inline]
    fn get2d_slice_uncheck(&self, position:[usize;2], length: usize) -> Option<&[T]> {
        let tp = self.index2d_uncheck(position).unwrap();
        Some(&self.data[tp..tp+length])
    }
}