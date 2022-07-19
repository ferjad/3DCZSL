import numpy as np
import ctypes as ct
import cv2
import sys
from eulerangles import euler2mat
import os

showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True

def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz)
    mousey = y / float(showsz)
    changed = True

cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onmouse)

import os.path
import ctypes
dll_name = "render_balls_so.so"
dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
dll = ctypes.CDLL(dllabspath)

#dll = np.ctypeslib.load_library('render_balls_so', '.')

def showpoints(xyz,c_gt=None, c_pred = None, waittime=0,
    showrot=False, magnifyBlue=0, freezerot=False, background=(255,255,255),
    normalizecolor=True, ballradius=10, id='0'):
    global showsz, mousex, mousey, zoom, changed
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*2.2)/showsz
    if c_gt is None:
        c0 = np.zeros((len(xyz), ), dtype='float32') + 255
        c1 = np.zeros((len(xyz), ), dtype='float32') + 255
        c2 = np.zeros((len(xyz), ), dtype='float32') + 255
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0


    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat = rotmat.dot(
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(xangle), -np.sin(xangle)],
                [0.0, np.sin(xangle), np.cos(xangle)],
            ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(
            np.array([
                [np.cos(yangle), 0.0, -np.sin(yangle)],
                [0.0, 1.0, 0.0],
                [np.sin(yangle), 0.0, np.cos(yangle)],
            ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius))

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)),
                        (30, showsz - 30), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)),
                        (30, showsz - 50), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0,
                        0.5, cv2.cv.CV_RGB(255, 0, 0))

    def render_withangle(xrot=0, yrot=0, zrot=0):
        rotmat = euler2mat(zrot, yrot, xrot)
        # rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius))

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)),
                        (30, showsz - 30), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)),
                        (30, showsz - 50), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0,
                        0.5, cv2.cv.CV_RGB(255, 0, 0))
    changed = True
    selection = 'gt_'
    while True:
        if changed:
            #render()
            #render_withangle(zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
            #render_withangle(zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
            render_withangle(zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
            changed = False
        #if cmd == ord('u'):
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                selection = 'gt_'
                if c_gt is None:
                    c0 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c1 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c2 = np.zeros((len(xyz), ), dtype='float32') + 255
                else:
                    c0 = c_gt[:, 0]
                    c1 = c_gt[:, 1]
                    c2 = c_gt[:, 2]
            else:
                selection = 'pred_'
                if c_pred is None:
                    c0 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c1 = np.zeros((len(xyz), ), dtype='float32') + 255
                    c2 = np.zeros((len(xyz), ), dtype='float32') + 255
                else:
                    c0 = c_pred[:, 0]
                    c1 = c_pred[:, 1]
                    c2 = c_pred[:, 2]
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            c0 = np.require(c0, 'float32', 'C')
            c1 = np.require(c1, 'float32', 'C')
            c2 = np.require(c2, 'float32', 'C')
            changed = True

        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        if cmd==ord('s'):
            cv2.imwrite('show3d_'+selection+id+'.png',show)
        if waittime!=0:
            break
    return cmd

def draw_three_pointclouds(xyz, c_gt=None, c_pred=None, magnifyBlue=0, background=(255,255,255), normalizecolor=True,
                           ballradius=10, id='0', topn_ind=None, filename='', c_err=None, iou_val=None,
                           gt_class_name=None, pred_class_name=None, pred_class_names=None, pred_biased_class_names=None, pred_labels=None, gt_labels=None):
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename = filename  + id + '/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    if topn_ind is not None:
        filename = filename + 'top_'+str(topn_ind)+'_'

    global showsz
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render_withangle(xrot=0, yrot=0, zrot=0):
        rotmat = euler2mat(zrot, yrot, xrot)
        # rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius))

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))

    def add_colorcode(lbls):

        if not lbls:
            return

        ctr = 0
        for lbl, lbl_i in lbls.items(): # lbl_i keeps the first occurence of label within the point cloud, so that
                                        # we can get the normalized color value from the recalculated cmap
            y = 40 + 30 * ctr
            ctr += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, y)
            fontScale = 1
            cc0 = int(c0[lbl_i])
            cc1 = int(c1[lbl_i])
            cc2 = int(c2[lbl_i])
            fontColor = (cc2, cc0, cc1) # GRB -> BGR
            #fontColor = (clr[2], clr[0], clr[1])  # GRB -> BGR
            lineType = 2
            cv2.putText(show, lbl,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    def add_class_name(class_name):
        if class_name is None:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 780)
        fontScale = 1
        fontColor = (0, 0, 0)  # GRB -> BGR
        lineType = 2

        cv2.putText(show, "Class: "+class_name,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def add_class_pred(pred_class_names, pred_biased_class_names):

        if pred_class_names == None:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 760)
        fontScale = 0.7
        fontColor = (0, 0, 0)  # GRB -> BGR
        lineType = 2

        cv2.putText(show, "Pred class: " + " ".join(pred_class_names),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        bottomLeftCornerOfText = (10, 780)

        cv2.putText(show, "Pred biased class: " + " ".join(pred_biased_class_names),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    if topn_ind is None:
        ### Draw GT point clouds
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]
        if normalizecolor:
            c0 /= (c0.max() + 1e-14) / 255.0
            c1 /= (c1.max() + 1e-14) / 255.0
            c2 /= (c2.max() + 1e-14) / 255.0
        c0 = np.require(c0, 'float32', 'C')
        c1 = np.require(c1, 'float32', 'C')
        c2 = np.require(c2, 'float32', 'C')

        render_withangle(zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        add_colorcode(gt_labels)
        add_class_name(gt_class_name)
        cv2.imwrite(filename + 'gt1.png', show)
        render_withangle(zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        add_colorcode(gt_labels)
        add_class_name(gt_class_name)
        cv2.imwrite(filename + 'gt2.png', show)
        render_withangle(zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        add_colorcode(gt_labels)
        add_class_name(gt_class_name)
        cv2.imwrite(filename + 'gt3.png', show)

    #print("GT saved to: ", filename)

    ### Draw predicted point clouds
    c0 = c_pred[:, 0]
    c1 = c_pred[:, 1]
    c2 = c_pred[:, 2]
    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0
    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    render_withangle(zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    add_colorcode(pred_labels)
    add_class_name(pred_class_name)
    cv2.imwrite(filename +  'pred1.png', show)
    render_withangle(zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    add_colorcode(pred_labels)
    add_class_name(pred_class_name)
    cv2.imwrite(filename + 'pred2.png', show)
    render_withangle(zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    add_colorcode(pred_labels)
    add_class_name(pred_class_name)
    cv2.imwrite(filename + 'pred3.png', show)

    #print("Pred saved to: ", filename)

    ### Draw error maps
    if c_err is not None:
        c0 = c_err[:, 0]
        c1 = c_err[:, 1]
        c2 = c_err[:, 2]
        c0 = np.require(c0, 'float32', 'C')
        c1 = np.require(c1, 'float32', 'C')
        c2 = np.require(c2, 'float32', 'C')

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 40)
        fontScale = 1
        fontColor = (204, 0, 204)
        lineType = 2

        render_withangle(zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        cv2.putText(show, "Seg IoU: "+ str(iou_val),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        add_class_pred(pred_class_names, pred_biased_class_names)
        cv2.imwrite(filename + 'pred1_err.png', show)
        render_withangle(zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        cv2.putText(show, "Seg IoU: " + str(iou_val),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        add_class_pred(pred_class_names, pred_biased_class_names)
        cv2.imwrite(filename +  'pred2_err.png', show)
        render_withangle(zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
        cv2.putText(show, "Seg IoU: " + str(iou_val),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        add_class_pred(pred_class_names, pred_biased_class_names)
        cv2.imwrite(filename + 'pred3_err.png', show)

        #print("Error map saved to: ", filename)

def draw_three_pointclouds_seg(xyz, c_map=None, magnifyBlue=0, background=(255,255,255), ballradius=10,
                            filename='', class_name=None,  labels=None, etiquette='', iou_val=None):

    global showsz
    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render_withangle(xrot=0, yrot=0, zrot=0):
        rotmat = euler2mat(zrot, yrot, xrot)
        # rotmat *= zoom
        nxyz = xyz.dot(rotmat) + [showsz / 2, showsz / 2, 0]

        ixyz = nxyz.astype('int32')
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]), ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p), ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p), c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p), c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius))

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(
                show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0],
                                           np.roll(show[:, :, 0], -1, axis=1))

    def add_colorcode(lbls):

        if not lbls:
            return

        ctr = 0
        for lbl, lbl_i in lbls.items(): # lbl_i keeps the first occurence of label within the point cloud, so that
                                        # we can get the normalized color value from the recalculated cmap
            y = 40 + 30 * ctr
            ctr += 1
            font = cv2.FONT_HERSHEY_TRIPLEX
            bottomLeftCornerOfText = (10, y)
            fontScale = 1
            cc0 = int(c0[lbl_i])
            cc1 = int(c1[lbl_i])
            cc2 = int(c2[lbl_i])
            fontColor = (cc2, cc0, cc1) # GRB -> BGR
            #fontColor = (clr[2], clr[0], clr[1])  # GRB -> BGR
            lineType = 2
            cv2.putText(show, lbl,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    def add_class_name(class_name):
        if class_name is None:
            return

        font = cv2.FONT_HERSHEY_TRIPLEX
        bottomLeftCornerOfText = (10, 780)
        fontScale = 1
        fontColor = (0, 0, 0)  # GRB -> BGR
        lineType = 2

        cv2.putText(show, "Class: "+class_name,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def add_etiquette(etiquette):
        if etiquette == '':
            return

        font = cv2.FONT_HERSHEY_TRIPLEX
        bottomLeftCornerOfText = (500, 780)
        fontScale = 1
        fontColor = (0, 0, 0)  # GRB -> BGR
        lineType = 2

        cv2.putText(show, etiquette,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def add_iou(iou_val):

        if iou_val == None:
            return

        font = cv2.FONT_HERSHEY_TRIPLEX
        bottomLeftCornerOfText = (500, 40)
        fontScale = 1
        fontColor = (0, 0, 0)  # GRB -> BGR
        lineType = 2

        cv2.putText(show, "Seg IoU: " + str(iou_val),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    ### Draw point clouds
    c0 = c_map[:, 0]
    c1 = c_map[:, 1]
    c2 = c_map[:, 2]
    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    render_withangle(zrot=270 / 180.0 * np.pi, xrot=350 / 180.0 * np.pi, yrot=20 / 180.0 * np.pi)
    add_colorcode(labels)
    add_class_name(class_name)
    add_etiquette(etiquette)
    add_iou(iou_val)
    cv2.imwrite(filename + '0.png', show)
    render_withangle(zrot=270 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=135 / 180.0 * np.pi)
    add_colorcode(labels)
    add_class_name(class_name)
    add_etiquette(etiquette)
    add_iou(iou_val)
    cv2.imwrite(filename + '1.png', show)
    render_withangle(zrot=270 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=315 / 180.0 * np.pi)
    add_colorcode(labels)
    add_class_name(class_name)
    add_etiquette(etiquette)
    add_iou(iou_val)
    cv2.imwrite(filename + '2.png', show)



if __name__ == '__main__':
    np.random.seed(100)
    showpoints(np.random.randn(2500, 3))