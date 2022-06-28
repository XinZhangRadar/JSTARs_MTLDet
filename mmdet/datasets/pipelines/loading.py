import os.path as osp
import os
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from PIL import Image

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

from .HTL.canny import edge_gen
from .HTL.density_gen import denisity,save_density_map
from .HTL.pot import compute_prob
import pdb
import cv2
from math import sqrt
from math import cos
from math import sin
import math
 
 
 

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        #cv2.imwrite()
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        #self.with_den = with_den
        #self.with_edge = with_edge

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        #import pdb;pdb.set_trace()
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        #print(results['gt_semantic_seg'].shape)
        results['seg_fields'].append('gt_semantic_seg')
        return results


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        
        if self.with_seg:
            results = self._load_semantic_seg(results)
            
        # if self.with_den:
        #     results = self._load
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg})'
        repr_str += f'poly2mask={self.poly2mask})'
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals(object):
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'

@PIPELINES.register_module()
class Load_view_Annotations(object):
    """Load mutiple types of annotations.

    Args:
        with_view (bool): Whether to parse and load the denisity annotation.
             Default: True.
        with_view (bool): Whether to parse and load the edge annotation.
            Default: True.
    """

    def __init__(self,with_view = True):

        self.with_view = with_view
        self.view_dict = {'1030010003D22F00':7,'1030010003993E00':6,'1030010002B7D800':5,'1030010002649200':4,'1030010003127500':3,'103001000307D800':2,'1030010003315300':1,
        '103001000392F600':0,'10300100023BC100':8,'1030010003CAF100':9,'10300100039AB000':10,'1030010003C92000':11,'103001000352C200':12,'1030010003472200':13,'10300100036D5200':14,
        '1030010003697400':15,'1030010003895500':16,'1030010003832800':17,'10300100035D1B00':18,'1030010003CCD700':19,'1030010003713C00':20,'10300100033C5200':21,'1030010003492700':22,
        '10300100039E6200':23,'1030010003BDDC00':24,'1030010003CD4300':25,'1030010003193D00':25}
        self.view_list = (-32,-29,-25,-21,-16,-13,-10,-7,8,10,14,19,23,27,30,34,36,39,42,44,46,47,49,50,52,53)

    def rad(self,d):
        return d * math.pi / 180.0
     
    def getDistance(self,lat1, lng1, lat2, lng2):
        EARTH_REDIUS = 1
        radLat1 = self.rad(lat1)
        radLat2 = self.rad(lat2)
        a = radLat1 - radLat2
        b = self.rad(lng1) - self.rad(lng2)
        s = 2 * math.asin(math.sqrt(math.pow(sin(a/2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b/2), 2)))
        s = s * EARTH_REDIUS
        #print("distance=",s)
        return s

    def _load_view(self, results):
        """Private function to load semantic denisity annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded denisity  annotations.
        """
        filename = results['filename']
        _view_id = filename.split('/')[4].split('_')[4]
        view, distance_gt = self.view_fliter(_view_id)
        print(view)
        print(distance_gt)


        results['distance_gt'] = distance_gt

        return results
    def view_fliter(self, _view_id):
        '''
            input: poses[ix,:]  = [distance,azimuth,elevation]
            return view[ix] = view ->[1,32]
        '''
        num_view = 26+1
        #num_view = 1+1
        #num_view = 1+1
        

        distance = np.zeros((num_view),dtype=np.float32)
 

        view, distance = self.view_label(_view_id,num_view)
        #pdb.set_trace()
        return view, distance
    def view_label(self, _view_id,num_view):

        view_label = self.view_dict[_view_id]
        azimuth = 0
        elevation = self.view_list[view_label]

        distance = np.zeros(num_view,dtype=np.float32)

        for view_int in range(len(self.view_list)) :
            
            mid_e = self.view_list[view_int]
            mid_a = 0
            distance[view_int] = self.getDistance(mid_e,mid_a,elevation,azimuth)
        
        #pdb.set_trace()
        distance[-1]=view_label
        return view_label, distance        

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded denisity, edge annotations.
        """


        
        if self.with_view:
            results = self._load_view(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_view={self.with_view}, '
        return repr_str



@PIPELINES.register_module()
class Load_HTC_Annotations(object):
    """Load mutiple types of annotations.

    Args:
        with_den (bool): Whether to parse and load the denisity annotation.
             Default: True.
        with_edge (bool): Whether to parse and load the edge annotation.
            Default: True.
    """

    def __init__(self,with_den = True, with_edge = True):

        self.with_den = with_den
        self.with_edge = with_edge

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        return results


    def _load_den(self, results):
        """Private function to load semantic denisity annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: The dict contains loaded denisity  annotations.
        """
        ori_img = results['after_flip_img']
        ori_size = ori_img.shape[0]
        #print(ori_size)
        bboxes = results['gt_bboxes']
        den = compute_prob(ori_size,ori_size,bboxes)
        '''
        bboxes = results['gt_bboxes'] * (128/ ori_size ) 
        den = compute_prob(128,128,bboxes)
        '''
        #image = results['img']

        #height, width, _ = image.shape





        #annPoints = np.zeros((bboxes.shape[0],2))

        #annPoints[:,0] = (bboxes[:,2] + bboxes[:,0])/2
        #annPoints[:,1] = (bboxes[:,3] + bboxes[:,1])/2

        #den, gt_count = denisity(height,width,annPoints,K = 3)
        save_density_map(den)

        results['gt_den'] = den

        return results
        

    def _load_edge(self, results):
        """Private function to load edge annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded edge annotations.
        """
        #edge_save_name = '/share/home/zhangxin/data/HRSID/true_edge/' + results['filename'].split('/')[-1].split('.')[0] + '.png' # true edge
        edge_save_name = '/share/home/zhangxin/mmdetection/data/coco_hrsid/edge_ori/' + results['filename'].split('/')[-1].split('.')[0] + '_edge.jpg' # canny edge
        #edge_save_name = '/share/home/zhangxin/data/LS-SSDD-v1.0-OPEN/edge_ori/' + results['filename'].split('/')[-1].split('.')[0] + '_edge.jpg'# LSSDD_edge
   

        ori_img = results['filename']
        img = cv2.imread(ori_img)
        cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/img.jpg',img)

        
        # if os.path.exists(edge_save_name): 
        #     print('1')
        #     results['gt_edge'] = cv2.imread(edge_save_name)[:,:,0]
        #     if results['if_scale']:
        #         results['gt_edge'] = mmcv.imrescale(results['gt_edge'], results['scale'], interpolation='nearest', backend='cv2')
        #     if results['flip']:
        #         results['gt_edge'] = mmcv.imflip(results['gt_edge'], direction=results['flip_direction'])      
        #     if results['if_pad']:
        #         results['gt_edge'] = mmcv.impad(results['gt_edge'] , shape=results['pad_shape'][:2])
        #     cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/edge.jpg',results['gt_edge']*255)     
        # else:
        #     print('2')
        #     bboxes = results['gt_bboxes']
        #     ori_img = results['filename']
        #     img = cv2.imread(ori_img)
        #     #ori_size = ori_img.shape[0]
        #     #ori_img = cv2.resize(ori_img, (128, 128))
   
        #     #bboxes =  bboxes * (128/ ori_size )  
        #     edge,edge_mask,mask = edge_gen(img,bboxes)
        #     results['gt_edge'] = edge_mask
        #     cv2.imwrite(edge_save_name,results['gt_edge'])
        #     if results['if_scale']:
        #         results['gt_edge'] = mmcv.imrescale(results['gt_edge'], results['scale'], interpolation='nearest', backend='cv2')
        #     if results['flip']:
        #         results['gt_edge'] = mmcv.imflip(results['gt_edge'], direction=results['flip_direction'])      
        #     if results['if_pad']:
        #         results['gt_edge'] = mmcv.impad(results['gt_edge'] , shape=results['pad_shape'][:2]) 
        #     cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/edge.jpg',results['gt_edge']*255)
        #     cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/edge_ori.jpg',results['gt_edge']*255)
        #     cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/edge_ori.jpg',results['gt_edge']*255)



        bboxes = results['gt_bboxes']
        print(results['filename'])
        print(bboxes)  
        edge,edge_mask,mask = edge_gen(img,bboxes)
        results['gt_edge'] = edge_mask
        cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/'+results['filename'].split('/')[-1].split('.')[0] + '_edge.jpg',edge*255)
        cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/'+results['filename'].split('/')[-1].split('.')[0] + '_edgemask.jpg',edge_mask*255)
        cv2.imwrite('/share/home/zhangxin/mmdetection/mmdet/datasets/pipelines/HTL/'+results['filename'].split('/')[-1].split('.')[0] + '_mask.jpg',mask*255)        

        #cv2.imwrite('/share/home/zhangxin/data/HRSID/edge128/ori.jpg',ori_img)
        #cv2.imwrite('/share/home/zhangxin/data/HRSID/edge128/canny.jpg',edge*255)        
        #cv2.imwrite('/share/home/zhangxin/data/HRSID/edge128/edge1.jpg',edge_mask*255)
        #cv2.imwrite('/share/home/zhangxin/mmdetection/edge.jpg',results['gt_edge']*255)
        

        return results
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded denisity, edge annotations.
        """


        
        if self.with_edge:
            results = self._load_edge(results)
        if self.with_den:
            results = self._load_den(results)
        #print(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_den={self.with_den}, '
        repr_str += f'with_edge={self.with_edge}, '
        return repr_str