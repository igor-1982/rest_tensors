use std::{fmt::Display, ops::{IndexMut,Index}, slice::SliceIndex};


//use crate::{ERIFull, ERIFold4, MatrixFull, MatrixFullSliceMut, MatrixFullSlice, MatrixUpperSliceMut, MatrixUpper, MatrixUpperSlice, RIFull, TensorOpt};

use crate::*;
//use crate::matrix::*;
//use crate::matrix::matrixfull::*;
//use crate::matrix::matrixfullslice::*;
//use crate::matrix::matrixupper::*;


fn contain_of(a:&[usize],b:&[usize]) -> bool {
    a.iter().zip(b.iter()).fold(true, |flg,(aa,bb)| flg && bb<aa)
}


// For the generic struct Tensors and its borrowed varians
pub trait Indexing {
    fn indexing(&self, position:&[usize]) -> usize;
    fn indexing_mat(&self, position:&[usize]) -> usize;
    fn reverse_indexing(&self, position:usize) -> Vec<usize>;
}
pub trait IndexingHP {
    fn indexing_last2rank(&self, position:&[usize]) -> usize;
}
//========================================================


/// TODO:: at present, all TensorIndex traits do not make the bound check for each dimension.
///        it is extremely danger, which should be addressed SOON!!!!!!

pub trait TensorIndex {
    // Indexing for regulear tensors, for example, ERIFull, MatrixFull
    fn index1d(&self, position:usize) -> Option<usize> {None}
    fn index2d(&self, position:[usize;2]) -> Option<usize> {None}
    fn index3d(&self, position:[usize;3]) -> Option<usize> {None}
    fn index4d(&self, position:[usize;4]) -> Option<usize> {None}
}

pub trait TensorIndexUncheck {
    // Indexing for the tensors with the elements in the upper block
    fn index2d_uncheck(&self, position:[usize;2]) -> Option<usize> {None}
    fn index4d_uncheck(&self, position:[usize;4]) -> Option<usize> {None}
}


impl <T> TensorIndex for ERIFull<T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index4d(&self, position: [usize;4]) -> Option<usize>
    {
        if contain_of(&self.size, &position) {
            Some(position.iter()
                .zip(self.indicing.iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}

impl <T> TensorIndex for ERIFold4<T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position: [usize;2]) -> Option<usize>
    {
        if contain_of(&self.size, &position) {
            Some(position.iter()
                .zip(self.indicing.iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
    #[inline]
    fn index4d(&self, position: [usize;4]) -> Option<usize>
    {
        let mut tp:[usize;2] = [0;2];
        tp[0] = if position[0] <= position[1] {
                   position[1]*(position[1]+1)/2+position[0]
                } else {
                   position[0]*(position[0]+1)/2+position[1]
                };
        tp[1] = if position[2] <= position[3] {
                   position[3]*(position[3]+1)/2+position[2]
                } else {
                   position[2]*(position[2]+1)/2+position[3]
                };
        if contain_of(&self.size, &tp) {
            Some(tp.iter()
                .zip(self.indicing.iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}
impl <T> TensorIndexUncheck for ERIFold4<T> {
    fn index4d_uncheck(&self, position:[usize;4]) -> Option<usize> {
        let rp = [(position[1]+1)*position[1]/2+position[0],
                            (position[3]+1)*position[3]/2+position[2]];
        if contain_of(&self.size,&rp) {
            Some(rp.iter()
                .zip(self.indicing.iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}

/// Indexing for MatrixFull and its (mut) borrowed variants.
///   MatrixFullSlice and MatrixFullMut
impl <T> TensorIndex for MatrixFull<T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        if contain_of(&self.size,&position) {
            Some(position.iter()
                .zip(self.indicing.iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}

impl <'a, T> TensorIndex for MatrixFullSliceMut<'a,T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        if contain_of(&self.size[..2],&position) {
            Some(position.iter()
                .zip(self.indicing[..2].iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}

impl <'a, T> TensorIndex for MatrixFullSlice<'a,T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        if contain_of(&self.size[..2],&position) {
            Some(position.iter()
                .zip(self.indicing[..2].iter())
                .map(|(pi,interval)| pi*interval)
                .sum())
        } else {
            None
        }
    }
}

/// Indexing for MatrixUpper and its (mut) borrowed variants.
///   MatrixUpperSlice and MatrixUpperMut
impl <T> TensorIndex for MatrixUpper<T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        let (i,j) = if position[0] <= position[1] {(position[0],position[1])} else {(position[1],position[0])};
        let tp = (j+1)*j/2+i;
        if tp < self.data.len() {Some(tp)} else {None}
    }
}
impl <T> TensorIndexUncheck for MatrixUpper<T> {
    #[inline]
    fn index2d_uncheck(&self, position:[usize;2]) -> Option<usize> {
        let tp = (position[1]+1)*position[1]/2+position[0];
        if tp < self.data.len() {Some(tp)} else {None}
    }
}
impl <'a, T> TensorIndex for MatrixUpperSliceMut<'a,T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        let (i,j) = if position[0] <= position[1] {(position[0],position[1])} else {(position[1],position[0])};
        let tp = (j+1)*j/2+i;
        if tp < self.data.len() {Some(tp)} else {None}
    }
}
impl <'a, T> TensorIndexUncheck for MatrixUpperSliceMut<'a,T> {
    #[inline]
    fn index2d_uncheck(&self, position:[usize;2]) -> Option<usize> {
        let tp = (position[1]+1)*position[1]/2+position[0];
        if tp < self.data.len() {Some(tp)} else {None}
    }
}
impl <'a, T> TensorIndex for MatrixUpperSlice<'a,T> {
    #[inline]
    fn index1d(&self, position: usize) -> Option<usize>
    {
        if position<self.data.len() {
            Some(position)
        }  else {
            None
        }
    }
    #[inline]
    fn index2d(&self, position:[usize;2]) -> Option<usize>
    {
        let (i,j) = if position[0] <= position[1] {(position[0],position[1])} else {(position[1],position[0])};
        let tp = (j+1)*j/2+i;
        if tp < self.data.len() {Some(tp)} else {None}
    }

    fn index3d(&self, position:[usize;3]) -> Option<usize> {None}

    fn index4d(&self, position:[usize;4]) -> Option<usize> {None}
}
impl <'a, T> TensorIndexUncheck for MatrixUpperSlice<'a,T> {
    #[inline]
    fn index2d_uncheck(&self, position:[usize;2]) -> Option<usize> {
        let tp = (position[1]+1)*position[1]/2+position[0];
        if tp < self.data.len() {Some(tp)} else {None}
    }
}

