import { useEffect, useState, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';

interface UploadResult {
  jobId: string;
  fileName: string;
  fileSize: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  status: string;
}

function formatFileSize(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

const ALLOWED_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'm4v', '3gp', 'mpeg', 'mpg', 'ts'];

function HomePage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [serverStatus, setServerStatus] = useState<'loading' | 'ok' | 'error'>('loading');
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadState, setUploadState] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    fetch('/api/health', { signal: controller.signal })
      .then((res) => res.json())
      .then((data: { status?: string }) => {
        setServerStatus(data.status === 'ok' ? 'ok' : 'error');
      })
      .catch(() => {
        setServerStatus('error');
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, []);

  const validateFile = useCallback((file: File): boolean => {
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setErrorMessage(t('upload.invalidFormat'));
      return false;
    }
    return true;
  }, [t]);

  const handleFile = useCallback((file: File) => {
    setErrorMessage('');
    if (validateFile(file)) {
      setSelectedFile(file);
      setUploadState('idle');
      setUploadProgress(0);
    }
  }, [validateFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const startUpload = useCallback(() => {
    if (!selectedFile) return;

    setUploadState('uploading');
    setUploadProgress(0);
    setErrorMessage('');

    const formData = new FormData();
    formData.append('video', selectedFile);

    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr;

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const result: UploadResult = JSON.parse(xhr.responseText);
        setUploadState('success');
        // Navigate to settings page with upload data
        setTimeout(() => {
          navigate(`/settings/${result.jobId}`, { state: result });
        }, 600);
      } else {
        setUploadState('error');
        try {
          const err = JSON.parse(xhr.responseText);
          setErrorMessage(err.error || t('upload.uploadError'));
        } catch {
          setErrorMessage(t('upload.uploadError'));
        }
      }
    });

    xhr.addEventListener('error', () => {
      setUploadState('error');
      setErrorMessage(t('upload.uploadError'));
    });

    xhr.open('POST', '/api/upload');
    xhr.send(formData);
  }, [selectedFile, navigate, t]);

  const cancelUpload = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    setUploadState('idle');
    setUploadProgress(0);
    setSelectedFile(null);
  }, []);

  return (
    <div className="flex flex-col items-center gap-10">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white">{t('home.welcome')}</h2>
        <p className="mt-3 text-lg text-gray-400">{t('home.description')}</p>
      </div>

      {/* Upload Zone */}
      <div className="w-full max-w-xl">
        <div
          onClick={() => uploadState !== 'uploading' && fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-all ${
            dragActive
              ? 'border-blue-400 bg-blue-950/30'
              : 'border-gray-700 bg-gray-900 hover:border-gray-500 hover:bg-gray-900/80'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Upload icon */}
          <div className="mb-4 flex justify-center">
            <svg className="h-12 w-12 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>

          <p className="text-lg text-gray-300">
            {dragActive ? t('upload.dropzoneActive') : t('upload.dropzone')}
          </p>
        </div>

        {/* Selected file info */}
        {selectedFile && (
          <div className="mt-4 rounded-xl border border-gray-800 bg-gray-900 p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-white">{selectedFile.name}</p>
                <p className="text-sm text-gray-400">
                  {t('upload.fileSize')}: {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </div>

            {/* Progress bar */}
            {uploadState === 'uploading' && (
              <div className="mt-3">
                <div className="h-3 w-full overflow-hidden rounded-full bg-gray-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="mt-1 text-sm text-gray-400">
                  {t('upload.uploadProgress', { percent: uploadProgress })}
                </p>
              </div>
            )}

            {/* Success state */}
            {uploadState === 'success' && (
              <p className="mt-3 text-sm font-medium text-green-400">
                {t('upload.uploadSuccess')}
              </p>
            )}

            {/* Error state */}
            {uploadState === 'error' && (
              <p className="mt-3 text-sm font-medium text-red-400">
                {errorMessage || t('upload.uploadError')}
              </p>
            )}

            {/* Action buttons */}
            <div className="mt-4 flex gap-3">
              {uploadState === 'idle' && (
                <button
                  onClick={startUpload}
                  className="rounded-xl bg-blue-600 px-6 py-2.5 font-semibold text-white transition-colors hover:bg-blue-500"
                >
                  {t('home.uploadButton')}
                </button>
              )}
              {uploadState === 'uploading' && (
                <button
                  onClick={cancelUpload}
                  className="rounded-xl bg-gray-700 px-6 py-2.5 font-semibold text-white transition-colors hover:bg-gray-600"
                >
                  {t('upload.cancel')}
                </button>
              )}
              {uploadState === 'error' && (
                <button
                  onClick={startUpload}
                  className="rounded-xl bg-blue-600 px-6 py-2.5 font-semibold text-white transition-colors hover:bg-blue-500"
                >
                  {t('upload.retry')}
                </button>
              )}
            </div>
          </div>
        )}

        {/* Format error */}
        {errorMessage && !selectedFile && (
          <p className="mt-3 text-center text-sm text-red-400">{errorMessage}</p>
        )}
      </div>

      {/* Divider + Create from prompt */}
      <div className="flex flex-col items-center gap-4">
        <span className="text-gray-500">{t('home.or')}</span>
        <button className="rounded-xl bg-purple-600 px-8 py-4 text-lg font-semibold text-white transition-colors hover:bg-purple-500">
          {t('home.createButton')}
        </button>
      </div>

      {/* Server status */}
      <div className="rounded-xl border border-gray-800 bg-gray-900 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="text-gray-400">{t('home.status')}:</span>
          {serverStatus === 'loading' && (
            <span className="text-gray-500">...</span>
          )}
          {serverStatus === 'ok' && (
            <span className="flex items-center gap-2 text-green-400">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-400" />
              {t('home.statusOk')}
            </span>
          )}
          {serverStatus === 'error' && (
            <span className="flex items-center gap-2 text-red-400">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-400" />
              {t('home.statusError')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage;
