export default {
    '608': {
        modelPath: 'faceModel',
        feedShape: {
            fw: 608,
            fh: 608
        },
        outputShapes: {
            from: [19, 19, 25, 1],
            to: [19, 19, 5, 5]
        }
    },
    '320': {
        modelPath: 'facemodel320',
        feedShape: {
            fw: 320,
            fh: 320
        },
        outputShapes: {
            from: [10, 10, 25, 1],
            to: [10, 10, 5, 5]
        }
    },
    '320fused': {
        modelPath: 'facemodelfused',
        feedShape: {
            fw: 320,
            fh: 320
        },
        outputShapes: {
            from: [10, 10, 25, 1],
            to: [10, 10, 5, 5]
        }
    },
    'separate': {
        modelPath: 'separablemodel',
        feedShape: {
            fw: 320,
            fh: 320
        },
        outputShapes: {
            from: [10, 10, 25, 1],
            to: [10, 10, 5, 5]
        }
    }
};
